"""Stable, language-neutral error codes with English API fallbacks."""
MESSAGES = {
 'stale_action':'The table has advanced. Wait for the latest state and try again.',
 'illegal_action':'That action is not legal here.',
 'table_failed':'This table encountered an error. Please start a new table.',
 'table_missing':'This table has expired or does not exist. Please start a new table.',
 'unsupported_variant':'This ruleset is not supported.',
 'unsupported_policy':'This AI difficulty is not supported.',
 'model_unavailable':'The standard AI is unavailable. Please choose Beginner.',
 'engine_unavailable':'The riichi engine is not ready.',
 'table_full':'All tables are busy. Please try again shortly.',
 'round_starting':'The hand is being prepared. Please wait.',
 'round_active':'Finish the current hand first.',
 'invalid_request':'The request is not valid.',
 'unknown_action':'This operation is not supported.',
 'origin_mismatch':'The request origin does not match this table.',
 'not_found':'This page does not exist.',
 'server_error':'The service encountered an error. Please try again.',
}
class ClientError(ValueError):
 def __init__(self,code):
  self.code=code
  super().__init__(MESSAGES[code])
