use ball_tree::BallTree;
use pyo3::{pyclass, pymethods};

use crate::alt_az_coord::AltAzCoord;

#[pyclass]
pub struct SkyPixelQuery
{
    ball_tree: BallTree<AltAzCoord, usize>,
    pixels: Vec<Option<bool>>
}


#[pymethods]
impl SkyPixelQuery
{
    #[new]
    pub fn new(star_positions: Vec<AltAzCoord>, pixels: Vec<Option<bool>>) -> SkyPixelQuery
    {
        let ball_tree = BallTree::new(star_positions, (0..pixels.len()).collect());
        SkyPixelQuery { ball_tree, pixels: pixels }
    }

    pub fn set_pixels(&mut self, pixels: Vec<Option<bool>>)
    {
        self.pixels = pixels;
    }

    pub fn query_radius(&self, coordinate: AltAzCoord, radius: f64) -> Option<f64>
    {
        let mut query = self.ball_tree.query();

        let values: Vec<bool> = query.nn_within(&coordinate, radius)
            .map(|(_, _, index)| self.pixels.get(*index).unwrap())
            .filter(|value| value.is_some())
            .map(|value| value.unwrap())
            .collect();
        
        if values.len() == 0
        {
            return None;
        }
        
        let cloudy_pixels = values.iter().filter(|x| **x).count();
        Some((cloudy_pixels as f64)/(values.len() as f64))
    }

    pub fn mask_radius(&mut self, coordinate: AltAzCoord, radius: f64)
    {
        let mut query = self.ball_tree.query();

        for (_, _, index) in query.nn_within(&coordinate, radius)
        {
            self.pixels[*index] = None;
        }
    }
}

#[cfg(test)]
mod tests
{
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_query_radius()
    {
        let distance = PI/2.0;

        let alt_az_coords: Vec<AltAzCoord> = vec![AltAzCoord::new(PI/4.0, 0.0), AltAzCoord::new(PI/4.0, PI)];

        let pixels: Vec<Option<bool>> = vec![Some(true), Some(false)];

        let sky_pixels = SkyPixelQuery::new(alt_az_coords, pixels);

        assert_eq!(sky_pixels.query_radius(AltAzCoord::new(PI/2.0, 0.0), distance), Some(0.5));
    }

    #[test]
    fn test_query_radius_none()
    {
        let distance = PI/2.0;

        let alt_az_coords: Vec<AltAzCoord> = vec![AltAzCoord::new(PI/4.0, 0.0), AltAzCoord::new(PI/4.0, PI)];

        let pixels: Vec<Option<bool>> = vec![None, None];

        let sky_pixels = SkyPixelQuery::new(alt_az_coords, pixels);

        assert_eq!(sky_pixels.query_radius(AltAzCoord::new(PI/2.0, 0.0), distance), None);
    }

    #[test]
    fn test_mask_radius()
    {
        let distance = PI/2.0;
        let alt_az_coords: Vec<AltAzCoord> = vec![AltAzCoord::new(PI/4.0, 0.0), AltAzCoord::new(PI/4.0, PI)];

        let pixels: Vec<Option<bool>> = vec![Some(true), Some(false)];

        let mut sky_pixels = SkyPixelQuery::new(alt_az_coords, pixels);

        sky_pixels.mask_radius(AltAzCoord::new(PI/2.0, 0.0), distance);
        assert_eq!(sky_pixels.pixels, vec![None, None])
    }
}