//! Defensive danger tables (Track B / v5 obs features).
//!
//! Statistical deal-in ("放铳") rate table against an *accepted riichi*
//! opponent, learned offline from ~5.2M riichi-facing discards in the Tenhou
//! houou corpus (see `scripts/build_deal_in_table.py`). Keyed by
//! (tile value category, suji/nosuji, junme bucket). Genbutsu is *not* in this
//! table: a player furiten on a tile can never ron it, so its deal-in prob is
//! exactly 0 and is handled by the caller.
//!
//! The suji definition here MUST match the one used to build the table:
//! a number tile with value `v` in suit `s` is "suji" iff the opponent has
//! discarded a same-suit tile of value `v-3` or `v+3`.

/// `[value_cat][safety][junme_bucket]`
/// - value_cat: 0=1/9, 1=2/8, 2=3/7, 3=4/6, 4=5, 5=honor
/// - safety:    0=suji, 1=nosuji  (genbutsu handled separately as 0)
/// - junme:     0=(1-6), 1=(7-12), 2=(13+)
pub const DEAL_IN_TABLE: [[[f32; 3]; 2]; 6] = [
    // 1/9
    [[0.01138, 0.01169, 0.01133], [0.03773, 0.04662, 0.05098]],
    // 2/8
    [[0.02622, 0.02756, 0.02657], [0.04947, 0.06166, 0.06962]],
    // 3/7
    [[0.03714, 0.04053, 0.04005], [0.06029, 0.07557, 0.08426]],
    // 4/6
    [[0.03605, 0.04897, 0.05180], [0.08367, 0.10885, 0.11942]],
    // 5
    [[0.03898, 0.05169, 0.05495], [0.08222, 0.11420, 0.12795]],
    // honor (never suji in practice; suji row mirrors nosuji as a safe default)
    [[0.01131, 0.01103, 0.01179], [0.01131, 0.01103, 0.01179]],
];

/// Tile id (0..34; 0-8=m, 9-17=p, 18-26=s, 27-33=honors) -> value category.
#[inline]
pub const fn value_cat(tid: usize) -> usize {
    if tid >= 27 {
        return 5; // honor
    }
    match tid % 9 {
        0 | 8 => 0, // 1/9
        1 | 7 => 1, // 2/8
        2 | 6 => 2, // 3/7
        3 | 5 => 3, // 4/6
        _ => 4,     // 5
    }
}

/// Junme (turn count) -> bucket index. Matches build_deal_in_table.py.
#[inline]
pub const fn junme_bucket(turn: u8) -> usize {
    if turn <= 6 {
        0
    } else if turn <= 12 {
        1
    } else {
        2
    }
}

/// Deal-in probability against an accepted riichi for a non-genbutsu tile.
/// `is_suji`: whether the tile is suji vs this opponent.
#[inline]
pub fn deal_in_prob(tid: usize, is_suji: bool, turn: u8) -> f32 {
    let safety = if is_suji { 0 } else { 1 };
    DEAL_IN_TABLE[value_cat(tid)][safety][junme_bucket(turn)]
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn value_cat_is_correct() {
        assert_eq!(value_cat(0), 0); // 1m
        assert_eq!(value_cat(8), 0); // 9m
        assert_eq!(value_cat(4), 4); // 5m
        assert_eq!(value_cat(13), 4); // 5p (idx 13, 13%9=4 -> value 5)
        assert_eq!(value_cat(9), 0); // 1p
        assert_eq!(value_cat(27), 5); // E
        assert_eq!(value_cat(33), 5); // C
    }

    #[test]
    fn danger_gradient_matches_theory() {
        let turn = 10;
        // no-suji: middle tiles more dangerous than terminals than honors
        let p5 = deal_in_prob(4, false, turn); // 5m
        let p3 = deal_in_prob(2, false, turn); // 3m
        let p1 = deal_in_prob(0, false, turn); // 1m
        let ph = deal_in_prob(27, false, turn); // honor
        assert!(p5 > p3, "5 should be more dangerous than 3");
        assert!(p3 > p1, "3 should be more dangerous than 1");
        assert!(p1 > ph, "1 should be more dangerous than honor");
        // suji strictly safer than no-suji for the same middle tile
        assert!(deal_in_prob(4, true, turn) < p5, "suji 5 safer than nosuji 5");
        // later junme is more dangerous
        assert!(deal_in_prob(4, false, 15) > deal_in_prob(4, false, 3));
    }
}
