use std::f64::consts::PI;

use ball_tree::Point;
use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone, Debug)]
pub struct AltAzCoord
{
    #[pyo3(get, set)]
    pub alt: f64,

    #[pyo3(get, set)]
    pub az: f64
}

#[pymethods]
impl AltAzCoord
{
    #[new]
    pub fn new(alt: f64, az:f64) -> Self {
        AltAzCoord {alt, az}
    }
}

impl PartialEq for AltAzCoord {
    fn eq(&self, other: &Self) -> bool {
        (self.alt == other.alt) && (self.az == other.az)
    }
}

impl Point for AltAzCoord
{
    fn distance(&self, other: &Self) -> f64 {
        let hav: f64 = haversine(other.alt - self.alt) + f64::cos(other.alt)*f64::cos(self.alt)*haversine(other.az - self.az);
        inv_haversine(hav)
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {

        let v = to_cartesian(self.alt, self.az);
        let w = to_cartesian(other.alt, other.az);

        let y = great_circle(v, w, -d);

        let (new_alt, new_az) = to_alt_az(y);

        AltAzCoord::new(new_alt, new_az)
    }
}

fn haversine(angle: f64) -> f64
{
    (1.0 - f64::cos(angle))/2.0
}

fn inv_haversine(haversine: f64) -> f64
{
    f64::acos(1.0 - 2.0*haversine)
}

fn to_cartesian(alt: f64, az: f64) -> (f64, f64, f64)
{
    let x = f64::cos(alt) * f64::cos(az);
    let y = f64::cos(alt) * f64::sin(az);
    let z = f64::sin(alt);

    (x, y, z)
}

fn to_alt_az((x, y, z): (f64, f64, f64)) -> (f64, f64)
{
    let r = f64::sqrt(x*x + y*y + z*z);
    let alt = PI/2.0 - f64::acos(z/r);
    let az = f64::atan2(y, x);

    (alt, az)
}

fn great_circle(v: (f64, f64, f64), w: (f64, f64, f64), t: f64) -> (f64, f64, f64)
{
    let theta = dot_product(v, w).acos();

    let y1 = great_circle_element(v.0, w.0, t, theta);
    let y2 = great_circle_element(v.1, w.1, t, theta);
    let y3 = great_circle_element(v.2, w.2, t, theta);

    (y1, y2, y3)
}

fn great_circle_element(v: f64, w:f64, t: f64, theta: f64) -> f64
{
    t.cos() * v + t.sin() * (1.0/theta.tan() * v - 1.0/theta.sin() * w)
}

fn dot_product((x1, y1, z1): (f64, f64, f64), (x2, y2, z2): (f64, f64, f64)) -> f64
{
    x1 * x2 + y1 * y2 + z1 * z2
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_equal()
    {
        let one = AltAzCoord::new(0.0, 0.0);
        let other = AltAzCoord::new(PI/2.0, 0.0);

        assert_eq!(one, one);
        assert_eq!(one, one.clone());

        assert_ne!(one, other);

    }

    #[test]
    fn test_distance_alt()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(PI/2.0, 0.0);

        assert!(from.distance(&from) < 1e-10);

        assert!(from.distance(&to) - PI/2.0 < 1e-10);
        assert!(to.distance(&from) - PI/2.0 < 1e-10);
    }

    #[test]
    fn test_distance_alt_eq()
    {
        let from = AltAzCoord::new(PI/2.0, 0.0);
        let to1 = AltAzCoord::new(PI/4.0, 0.0);
        let to2 = AltAzCoord::new(PI/4.0, PI);

        assert!(from.distance(&to2) - from.distance(&to1) < 1e-10);
    }

    #[test]
    fn test_distance_az()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(0.0, PI/4.0);

        assert!(from.distance(&to) - PI/4.0 < 1e-10);
        assert!(to.distance(&from) - PI/4.0 < 1e-10);
    }

    #[test]
    fn test_move_zero()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(PI/2.0, 0.0);

        let result = from.move_towards(&to, 0.0);

        assert!(result.alt < 1e-10);
        assert!(result.az  < 1e-10);
    }

    #[test]
    fn test_move_full_distance()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(PI/2.0, 0.0);

        let result = from.move_towards(&to, PI/2.0);

        assert!(result.alt - PI/2.0 < 1e-10);
        assert!(result.az  < 1e-10);
    }

    #[test]
    fn test_move_to_alt()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(PI/2.0, 0.0);

        let result = from.move_towards(&to, PI/4.0);

        assert!(result.alt - PI/4.0 < 1e-10);
        assert!(result.az - 0.0 < 1e-10);
    }

    #[test]
    fn test_move_to_az()
    {
        let from = AltAzCoord::new(0.0, 0.0);
        let to = AltAzCoord::new(0.0, PI/2.0);

        let result = to.move_towards(&from, PI/4.0);

        assert!(result.az - PI/4.0 < 1e-10);
        assert!(result.alt - 0.0 < 1e-10);
    }
}