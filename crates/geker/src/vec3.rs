// This file implementes the "vector 3d" class. Actually, we represent
// the three-dimensional entries as a plane fixed size array of `[f64; 3]`.
// This makes in the interpolation with libm easier.
//
// We use libm as it yields platform independent bit-same results (compared
// to the build-in math funtions that use the platform build in ones, which
// can different between platforms).
//
// We use methods as we will soon have these methods over different generic
// types.

pub type V3 = [f64; 3];

pub fn add(a: V3, b: V3) -> V3   { 
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]] 
}

pub fn sub(a: V3, b: V3) -> V3   { 
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0]*s, a[1]*s, a[2]*s] 
}

pub fn dot(a: V3, b: V3) -> f64  { 
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [a[1]*b[2] - a[2]*b[1],
     a[2]*b[0] - a[0]*b[2],
     a[0]*b[1] - a[1]*b[0]]
}

pub fn norm(a: V3) -> f64 { 
    libm::sqrt(dot(a, a)) 
}

#[cfg(test)] 
mod tests {
    use super::*;

    #[test] fn add_sub() {
        assert_eq!(add([1.,2.,3.], [4.,5.,6.]), [5.,7.,9.]);
        assert_eq!(sub([1.,2.,3.], [4.,5.,6.]), [-3.,-3.,-3.]);
    }
    #[test] fn scale_dot() {
        assert_eq!(scale([1.,-2.,3.], 2.), [2.,-4.,6.]);
        assert_eq!(dot([1.,2.,3.], [4.,5.,6.]), 32.);   // 4 + 10 + 18
    }
    #[test] fn cross_basis() {
        assert_eq!(cross([1.,0.,0.], [0.,1.,0.]), [0.,0.,1.]);   // x × y = z
        assert_eq!(cross([0.,1.,0.], [0.,0.,1.]), [1.,0.,0.]);   // y × z = x
        assert_eq!(cross([0.,1.,0.], [1.,0.,0.]), [0.,0.,-1.]);  // order matters
    }
    #[test] fn cross_general() {
        // (2,3,4) × (5,6,7) = (3*7-4*6, 4*5-2*7, 2*6-3*5) = (-3, 6, -3)
        assert_eq!(cross([2.,3.,4.], [5.,6.,7.]), [-3.,6.,-3.]);
        // the result is perpendicular to both inputs
        assert_eq!(dot([-3.,6.,-3.], [2.,3.,4.]), 0.);
    }
    #[test] fn norm_pythagorean() {
        assert_eq!(norm([3.,4.,0.]), 5.);
        assert_eq!(norm([2.,3.,6.]), 7.);   // 4 + 9 + 36 = 49
    }
}
