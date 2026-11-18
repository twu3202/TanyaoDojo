"""Small CPU web application. Run: python -m deploy.play.server (PORT=7860)."""
from __future__ import annotations
import copy
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]
STATIC=Path(__file__).parent/'static'
ASSETS=Path(os.environ.get('TANYAO_ASSETS',ROOT/'runs/mvp_assets/hf'))
NATIVE=Path(os.environ.get('TANYAO_NATIVE_DIR',ASSETS))
sys.path.insert(0,str(NATIVE))
from .registry import VARIANTS, SESSION_OPTIONS
from .errors import ClientError, MESSAGES
from .sichuan_game import SichuanSession
from .riichi_game import RiichiSession
from .hongzhong_game import HongzhongSession
from .suited_game import SuitedSession
from .honors_game import HonorsSession

VERSION='0.20.0'
FACTORIES={'riichi':RiichiSession,'sichuan':SichuanSession,'hongzhong':HongzhongSession,
           'suited-discard-win':SuitedSession,'continuous-win':SuitedSession,'honors-and-chows':HonorsSession,
           'honors-horse-tiles':HonorsSession,'ka-wu-xing':HonorsSession,'hangzhou-wildcard':HonorsSession}


class Manager:
    def __init__(self,models=None):
        self.models=models or {}
        self.sessions={}
        self.lock=threading.RLock()

    def get(self,sid):
        with self.lock:
            s=self.sessions.get(sid)
            if s is None:raise ClientError('table_missing')
            return s

    def create(self,variant,policy):
        if not isinstance(variant,str) or variant not in VARIANTS:raise ClientError('unsupported_variant')
        rules=VARIANTS[variant]
        if policy not in [entry['id'] for entry in rules['policies']]:raise ClientError('unsupported_policy')
        if policy=='model' and variant not in self.models:raise ClientError('model_unavailable')
        if variant=='riichi':
            try:import libriichi
            except ImportError as e:raise ClientError('engine_unavailable') from e
        with self.lock:
            for key,s in list(self.sessions.items()):
                if s.closed or time.monotonic()-s.last_seen>1800:
                    s.close();del self.sessions[key]
            if len(self.sessions)>=32:raise ClientError('table_full')
            cls=FACTORIES[rules['family_id']]
            s=cls(policy,model=self.models.get(variant),**SESSION_OPTIONS.get(variant,{}))
            s.rule_version=rules['rule_version']
            self.sessions[s.id]=s;s.begin()
            return s

    def catalog(self):
        variants=copy.deepcopy(list(VARIANTS.values()))
        for v in variants:
            for p in v['policies']:p['available']=p['id']!='model' or v['id'] in self.models
            if v['id'] not in self.models:v['default_policy']='heuristic'
        return {'variants':variants,'version':VERSION,'default_language':'en','languages':[{'id':'en','name':'English'},{'id':'zh-CN','name':'简体中文'}],'scope':'Playable, versioned rulesets. More families are being added.'}


MANAGER=Manager()


class Handler(BaseHTTPRequestHandler):
    server_version='TanyaoDojo/0.1'
    def log_message(self,fmt,*args):
        # Session capability URLs are intentionally not written to logs.
        if args and str(args[0]).startswith(('GET /api/game','POST /api/game')):return
        super().log_message(fmt,*args)

    def reply(self,status,data):
        body=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path=='/api/catalog':return self.reply(200,MANAGER.catalog())
            if path=='/health':return self.reply(200,{'status':'ok','version':VERSION,'default_language':'en','rulesets':list(VARIANTS),'models':sorted(MANAGER.models)})
            if path.startswith('/api/game/'):
                return self.reply(200,MANAGER.get(path.split('/')[3]).snapshot())
            rel='index.html' if path=='/' else path.lstrip('/')
            file=(STATIC/rel).resolve()
            if not file.is_relative_to(STATIC.resolve()) or not file.is_file():
                return self.reply(404,{'error':MESSAGES['not_found'],'error_code':'not_found'})
            data=file.read_bytes()
            self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-cache' if file.suffix in ('.html','.js','.css') else 'public, max-age=3600')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.end_headers();self.wfile.write(data)
        except ClientError as e:self.reply(404 if e.code=='table_missing' else 400,{'error':str(e),'error_code':e.code})
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                return self.reply(403,{'error':MESSAGES['origin_mismatch'],'error_code':'origin_mismatch'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ClientError('invalid_request')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ClientError('invalid_request')
            path=urlsplit(self.path).path
            if path=='/api/game':
                s=MANAGER.create(body.get('variant'),body.get('policy','heuristic'))
                return self.reply(201,s.snapshot())
            parts=path.strip('/').split('/')
            if len(parts)!=4 or parts[:2]!=['api','game']:return self.reply(404,{'error':MESSAGES['unknown_action'],'error_code':'unknown_action'})
            s=MANAGER.get(parts[2]);op=parts[3]
            if op=='action':s.act(body.get('revision'),str(body.get('choice')))
            elif op=='auto':s.auto(bool(body.get('enabled',True)))
            elif op=='next':s.begin()
            elif op=='leave':s.close()
            else:raise ClientError('unknown_action')
            return self.reply(200,s.snapshot())
        except ClientError as e:self.reply(404 if e.code=='table_missing' else 400,{'error':str(e),'error_code':e.code})
        except (ValueError,TypeError,json.JSONDecodeError) as e:self.reply(400,{'error':str(e),'error_code':'invalid_request'})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception:
            logging.exception('Request failed');self.reply(500,{'error':MESSAGES['server_error'],'error_code':'server_error'})


def main():
    from .policies import RiichiModel
    if (ASSETS/'model.pth').exists():MANAGER.models['riichi']=RiichiModel(ASSETS)
    from .sichuan_model import SichuanModel
    sc=Path(os.environ.get('TANYAO_SICHUAN_WEIGHTS',ASSETS/'sichuan.npz'))
    if sc.exists():MANAGER.models['sichuan']=SichuanModel(sc)
    address=os.environ.get('HOST','127.0.0.1');port=int(os.environ.get('PORT','7860'))
    server=ThreadingHTTPServer((address,port),Handler)
    print(f'TanyaoDojo ready http://{address}:{port} | models: {list(MANAGER.models)}',flush=True)
    try:server.serve_forever()
    finally:
        for session in MANAGER.sessions.values():session.close()
        server.server_close()


if __name__=='__main__':main()
