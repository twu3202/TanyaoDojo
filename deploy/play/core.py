from __future__ import annotations
import copy
import secrets
import threading
import time
from collections import deque

from .errors import ClientError, MESSAGES

NAMES = ['You', 'Bamboo', 'Dew', 'Pine']
SUITS = ['Characters', 'Dots', 'Bamboo']
TILES = [f'{n}{s}' for s in 'mps' for n in range(1, 10)] + list('ESWNPFC')


def tile_name(i):
    return TILES[i]


def tile_label(t):
    if t in 'ESWNPFC' and len(t) == 1:
        return dict(zip('ESWNPFC', ['East', 'South', 'West', 'North', 'White dragon', 'Green dragon', 'Red dragon']))[t]
    return ('Red ' if t.endswith('r') else '') + f'{t[0]} {dict(m="Characters",p="Dots",s="Bamboo")[t[1]]}'


def tile_sort(t):
    b = t.rstrip('r')
    return (TILES.index(b), not t.endswith('r'))


class Closed(Exception):
    pass


class GameSession:
    def __init__(self, variant, policy, *, seed=None, pacing=0.12):
        self.id = secrets.token_urlsafe(24)
        self.variant, self.policy = variant, policy
        self.rule_version = None
        self.seed = secrets.randbits(62) if seed is None else seed
        self.pacing = pacing
        self.cv = threading.Condition(threading.RLock())
        self.version = 0
        self.status = 'starting'
        self.closed = False
        self.automatic = False
        self.last_seen = time.monotonic()
        self.options = []
        self.answer = None
        self.preferred = None
        self.table = {}
        self.result = None
        self.error = None
        self.error_code = None
        self.messages = deque(maxlen=60)
        self._message_id = 0
        self.thread = None

    def log(self, text, kind='normal'):
        with self.cv:
            self._message_id += 1
            self.messages.append({'id': self._message_id, 'text': text, 'kind': kind})

    def log_event(self, code, params=None, kind='normal'):
        with self.cv:
            self._message_id += 1
            self.messages.append({'id':self._message_id,'code':code,'params':params or {},'kind':kind})

    def publish(self, table, status='thinking'):
        with self.cv:
            if self.closed:
                raise Closed()
            self.table = table
            self.status = status
            self.version += 1
            self.cv.notify_all()

    def ask(self, table, choices, preferred):
        """Only prevalidated choices are ever accepted from the browser."""
        with self.cv:
            if self.closed:
                raise Closed()
            self.table = table
            self.options = [{'id': str(i), **x} for i, x in enumerate(choices)]
            self.preferred = str(preferred)
            self.answer = None
            self.status = 'your_turn'
            self.version += 1
            self.cv.notify_all()
            until = time.monotonic()+1800
            while self.answer is None:
                if self.closed or time.monotonic() > until:
                    raise Closed()
                if self.automatic:
                    self.answer = self.preferred
                    break
                self.cv.wait(timeout=2)
            choice = self.options[int(self.answer)]
            self.options = []
            self.status = 'thinking'
            self.version += 1
            return choice['action']

    def act(self, revision, option_id):
        with self.cv:
            if self.status != 'your_turn' or revision != self.version or self.answer is not None:
                raise ClientError('stale_action')
            if option_id not in [x['id'] for x in self.options]:
                raise ClientError('illegal_action')
            self.answer = option_id
            self.version += 1
            self.status = 'thinking'
            self.cv.notify_all()

    def auto(self, enabled=True):
        with self.cv:
            self.automatic = enabled
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.status = 'closed'
            self.cv.notify_all()

    def start_worker(self, fn):
        def run():
            try:
                fn()
            except Closed:
                with self.cv:
                    self.status = 'closed'
                    self.closed = True
            except Exception:
                import logging
                if not self.closed:
                    logging.exception('Game worker failed: %s', self.id[:8])
                with self.cv:
                    self.status = 'closed' if self.closed else 'error'
                    self.error_code = None if self.closed else 'table_failed'
                    self.error = None if self.closed else MESSAGES['table_failed']
                    self.version += 1
            finally:
                with self.cv:
                    self.options = []
                    self.cv.notify_all()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.cv:
            self.last_seen = time.monotonic()
            return copy.deepcopy({'id': self.id, 'variant': self.variant, 'policy': self.policy,
                'rule_version': self.rule_version, 'revision': self.version, 'status': self.status, 'automatic': self.automatic,
                'table': self.table, 'choices': [{k:v for k,v in x.items() if k != 'action'} for x in self.options] if self.status == 'your_turn' else [],
                'messages': list(self.messages), 'result': self.result, 'error': self.error, 'error_code':self.error_code})
