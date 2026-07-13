//! Defensive danger tables for the v5 obs features (Track B).
//!
//! Indices are relative to the player: `[0]`=shimocha, `[1]`=toimen,
//! `[2]`=kamicha (i.e. relative opponents 1, 2, 3). All values are 0 for a
//! non-riichi opponent in v0 (riichi-only defense).

pub struct DefensiveTables {
    /// Per-opponent, per-tile deal-in probability in `[0, 1]`. 0 for genbutsu
    /// (furiten, cannot ron) and for non-riichi opponents.
    pub deal_in_prob: [[f32; 34]; 3],
    /// 1 if the tile is genbutsu (in that riichi opponent's kawa).
    pub genbutsu: [[bool; 34]; 3],
    /// 1 if the tile is suji vs that riichi opponent.
    pub suji: [[bool; 34]; 3],
}
