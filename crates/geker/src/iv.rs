// Closed interval [lo, hi] of f64 with outward rounding.
// Containment: if x ∈ X and y ∈ Y, then x op y ∈ X op Y, rounding error included.

use core::ops::{Add, Mul, Neg, Sub};


/// Widen by one ulp each way. The outward change is due do the loss
/// of precision during operations on floating point numbers.
fn out(lo: f64, hi: f64) -> Iv { Iv::new(lo.next_down(), hi.next_up()) }

fn min4(v: [f64; 4]) -> f64 { v[0].min(v[1]).min(v[2].min(v[3])) }
fn max4(v: [f64; 4]) -> f64 { v[0].max(v[1]).max(v[2].max(v[3])) }


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iv {
    pub lo: f64,
    pub hi: f64,
}

impl Iv {
    pub fn new(lo: f64, hi: f64) -> Iv {
        // Finite endpoints only: rules out NaN (which min/max would drop silently,
        // e.g. 0 × ∞) and makes an overflow fail loudly instead of giving a wrong bound.
        assert!(lo.is_finite() && hi.is_finite(), "non-finite interval [{lo}, {hi}]");
        assert!(lo <= hi, "inverted interval [{lo}, {hi}]");
        Iv { lo, hi }
    }

    pub fn point(x: f64) -> Iv { Iv::new(x, x) }
    pub fn contains(self, x: f64) -> bool { self.lo <= x && x <= self.hi }
    pub fn contains_zero(self) -> bool { self.contains(0.0) }
    pub fn width(self) -> f64 { self.hi - self.lo }

    /// 1/x. None = refused: x contains 0.
    pub fn recip(self) -> Option<Iv> {
        if self.contains_zero() { return None; }
        Some(out(1.0 / self.hi, 1.0 / self.lo))
    }

    /// x². Special handling if zero is contained in interval.
    pub fn sqr(self) -> Iv {
        let (a, b) = (self.lo.abs(), self.hi.abs());
        let (m, n) = (a.min(b), a.max(b));
        if self.contains_zero() {
            Iv::new(0.0, (n * n).next_up())   // lower bound 0 is exact
        } else {
            out(m * m, n * n)
        }
    }

    /// None = refused: the denominator contains 0, so the quotient is unbounded.
    pub fn div(self, o: Iv) -> Option<Iv> {
        if o.contains_zero() { return None; }
        let q = [self.lo/o.lo, self.lo/o.hi, self.hi/o.lo, self.hi/o.hi];
        Some(out(min4(q), max4(q)))
    }
}


// Adding operator overloading for +-, negation and multiply. No support
// for division as the result might be None.

impl Add for Iv {
    type Output = Iv;
    fn add(self, o: Iv) -> Iv { out(self.lo + o.lo, self.hi + o.hi) }
}

impl Sub for Iv {
    type Output = Iv;
    // Crossed endpoints: the smallest difference is lo − hi.
    fn sub(self, o: Iv) -> Iv { out(self.lo - o.hi, self.hi - o.lo) }
}

impl Neg for Iv {
    type Output = Iv;
    fn neg(self) -> Iv { Iv::new(-self.hi, -self.lo) }   // exact, no widening
}

impl Mul for Iv {
    type Output = Iv;
    fn mul(self, o: Iv) -> Iv {
        let p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi];
        out(min4(p), max4(p))
    }
}

#[cfg(test)] 
mod tests {
    use super::*;

    /// The expected result when the exact answer is [lo, hi]:
    /// every operation except neg widens by one ulp each way.
    fn w(lo: f64, hi: f64) -> Iv { Iv::new(lo.next_down(), hi.next_up()) }
    fn iv(lo: f64, hi: f64) -> Iv { Iv::new(lo, hi) }

    #[test] #[should_panic] 
    fn rejects_nan() { 
        Iv::new(f64::NAN, 1.0); 
    }
    
    #[test] #[should_panic] 
    fn rejects_inf() { 
        Iv::new(0.0, f64::INFINITY);
    }

    #[test] #[should_panic] 
    fn rejects_inverted() { 
        Iv::new(2.0, 1.0); 
    }
    
    #[test] #[should_panic] 
    fn overflow_panics()  { 
        let _ = Iv::point(f64::MAX) * Iv::point(2.0); 
    }
    
    #[test] 
    fn recip_refuses_zero() { 
        assert_eq!(Iv::new(-1.0, 1.0).recip(), None); 
    }

        // ---- basics ---------------------------------------------------------

    #[test]
    fn point_contains_width() {
        let p = Iv::point(2.5);
        assert_eq!((p.lo, p.hi, p.width()), (2.5, 2.5, 0.0));
        let x = iv(-1.0, 3.0);
        assert!(x.contains(-1.0) && x.contains(3.0) && x.contains(0.5)); // closed ends
        assert!(!x.contains((-1.0f64).next_down()) && !x.contains(3.0f64.next_up()));
        assert_eq!(x.width(), 4.0);
    }

    #[test]
    fn contains_zero_cases() {
        assert!(iv(-1.0, 1.0).contains_zero());
        assert!(iv(0.0, 1.0).contains_zero());       // zero at an endpoint counts
        assert!(iv(-1.0, 0.0).contains_zero());
        assert!(!iv(1e-300, 1.0).contains_zero());
        assert!(!iv(-1.0, -1e-300).contains_zero());
    }

    #[test]
    #[should_panic]
    fn widening_past_max_panics() {
        // f64::MAX + 0 is exact, but widening steps the upper bound to ∞.
        let _ = Iv::point(f64::MAX) + Iv::point(0.0);
    }

    // ---- add, sub, neg --------------------------------------------------

    #[test]
    fn add_endpoints() {
        assert_eq!(iv(1.0, 2.0) + iv(3.0, 5.0), w(4.0, 7.0));
        assert_eq!(iv(-2.0, 1.0) + iv(-1.0, 4.0), w(-3.0, 5.0));
    }

    #[test]
    fn sub_crosses_endpoints() {
        // [1,2] − [0,1] = [1−1, 2−0] = [0, 2], not [1, 1]
        assert_eq!(iv(1.0, 2.0) - iv(0.0, 1.0), w(0.0, 2.0));
        assert_eq!(iv(-1.0, 1.0) - iv(2.0, 5.0), w(-6.0, -1.0));
    }

    #[test]
    fn sub_self_is_not_zero() {
        // Intervals don't know both operands are the same number.
        let x = iv(1.0, 2.0);
        assert_eq!(x - x, w(-1.0, 1.0));
    }

    #[test]
    fn neg_is_exact() {
        assert_eq!(-iv(1.0, 3.0), iv(-3.0, -1.0));
        assert_eq!(-iv(-2.0, 5.0), iv(-5.0, 2.0));
    }

    #[test]
    fn add_rounding_is_enclosed() {
        // 1 + 1e-20 rounds to exactly 1.0 in f64; the true sum is above 1.
        assert!((Iv::point(1.0) + Iv::point(1e-20)).hi > 1.0);
        assert!((Iv::point(1.0) - Iv::point(1e-20)).lo < 1.0);
    }
    
    // ---- mul: every sign combination -------------------------------------

    #[test]
    fn mul_sign_cases() {
        let (pos, neg, mix) = (iv(3.0, 4.0), iv(-4.0, -3.0), iv(-1.0, 2.0));
        assert_eq!(iv(1.0, 2.0) * pos, w(3.0, 8.0));       // + · +
        assert_eq!(iv(-2.0, -1.0) * pos, w(-8.0, -3.0));   // − · +
        assert_eq!(iv(-2.0, -1.0) * neg, w(3.0, 8.0));     // − · −
        assert_eq!(mix * pos, w(-4.0, 8.0));               // ± · +
        assert_eq!(mix * neg, w(-8.0, 4.0));               // ± · −
        assert_eq!(mix * iv(-3.0, 1.0), w(-6.0, 3.0));     // ± · ±: products 3, −1, −6, 2
    }

    #[test]
    fn mul_commutes() {
        let xs = [iv(1.0, 2.0), iv(-2.0, -1.0), iv(-1.0, 2.0), iv(-3.0, 1.0), Iv::point(0.0)];
        for &a in &xs { for &b in &xs { assert_eq!(a * b, b * a); } }
    }

    #[test]
    fn mul_by_zero_point() {
        assert_eq!(Iv::point(0.0) * iv(-1.0, 2.0), w(0.0, 0.0));
    }

    // ---- sqr ------------------------------------------------------------

    #[test]
    fn sqr_cases() {
        assert_eq!(iv(2.0, 3.0).sqr(), w(4.0, 9.0));
        assert_eq!(iv(-3.0, -2.0).sqr(), w(4.0, 9.0));
        // Straddling zero: the lower bound is exactly 0, not widened below.
        assert_eq!(iv(-1.0, 2.0).sqr(), iv(0.0, 4.0f64.next_up()));
        assert_eq!(iv(-3.0, 2.0).sqr(), iv(0.0, 9.0f64.next_up()));
    }

    #[test]
    fn sqr_tighter_than_mul() {
        let x = iv(-1.0, 2.0);
        assert_eq!(x * x, w(-2.0, 4.0));   // x·x can't know it's a square
        assert_eq!(x.sqr().lo, 0.0);
    }

    // ---- recip, div -----------------------------------------------------

    #[test]
    fn recip_cases() {
        assert_eq!(iv(2.0, 4.0).recip(), Some(w(0.25, 0.5)));
        assert_eq!(iv(-4.0, -2.0).recip(), Some(w(-0.5, -0.25)));
        assert_eq!(iv(0.0, 1.0).recip(), None);    // zero at an endpoint refuses too
        assert_eq!(iv(-1.0, 0.0).recip(), None);
    }

    #[test]
    fn recip_rounding_is_enclosed() {
        let r = Iv::point(3.0).recip().unwrap();   // 1/3 isn't representable
        assert!(r.lo < 1.0 / 3.0 && 1.0 / 3.0 < r.hi);
    }

     #[test]
    fn div_cases() {
        assert_eq!(iv(1.0, 2.0).div(iv(4.0, 8.0)), Some(w(0.125, 0.5)));
        // quotients −0.5, −0.25, 1, 0.5
        assert_eq!(iv(-1.0, 2.0).div(iv(2.0, 4.0)), Some(w(-0.5, 1.0)));
        // quotients −0.25, −0.5, −0.5, −1
        assert_eq!(iv(1.0, 2.0).div(iv(-4.0, -2.0)), Some(w(-1.0, -0.25)));
    }

    #[test]
    fn div_refuses_zero_denominator() {
        assert_eq!(iv(1.0, 2.0).div(iv(-1.0, 1.0)), None);
        assert_eq!(iv(1.0, 2.0).div(iv(0.0, 1.0)), None);
        assert_eq!(iv(1.0, 2.0).div(Iv::point(0.0)), None);
    }   
}