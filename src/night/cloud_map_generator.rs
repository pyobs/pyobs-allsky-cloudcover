use std::f64::consts::PI;

use ball_tree::BallTree;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::night::alt_az_coord::AltAzCoord;
use crate::night::average::Average;
use crate::night::star::Star;
use crate::night::star_counter::StarCounter;
use crate::night::weighted_value::WeightedValue;

#[pyclass]
pub struct MagnitudeMapGenerator
{
    ball_tree: BallTree<AltAzCoord, Star>,
}

#[pymethods]
impl MagnitudeMapGenerator
{
    #[new]
    pub fn new(star_positions: Vec<AltAzCoord>, stars: Vec<Star>) -> MagnitudeMapGenerator
    {
        let ball_tree = BallTree::new(star_positions, stars);
        MagnitudeMapGenerator{ball_tree}
    }

    pub fn query(&self, position: AltAzCoord, distance: f64) -> Option<Average>
    {
        let mut query = self.ball_tree.query();

        let nn_result = query.nn_within(&position, distance);

        let mut star_counter = StarCounter::default();

        for (_, distance, star) in nn_result
        {
            let weighted_vmag = WeightedValue::new(star.get_v_mag(), -distance * PI / (7.0 * 180.0));
            if star.get_found()
            {
                star_counter.add_visible_v_mag(weighted_vmag)
            }else {
                star_counter.add_n_visible_v_mag(weighted_vmag)
            }
        }

        star_counter.calc_v_mag_border_value()
    }

    pub fn query_many(&self, positions: Vec<Option<AltAzCoord>>, distance: f64) -> Vec<Option<Average>>
    {
        positions.into_par_iter().map(|x| self.query_optional(x, distance)).collect()
    }
}

impl MagnitudeMapGenerator
{
    fn query_optional(&self, position: Option<AltAzCoord>, distance: f64) -> Option<Average>
    {
        match position
        {
            Some(pos) => self.query(pos, distance),
            None => None
        }
    }
}


#[cfg(test)]
mod tests
{
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_gen_cloud_map()
    {
        let distance = PI/2.0;

        let alt_az_coords: Vec<AltAzCoord> = vec![AltAzCoord::new(PI/4.0, 0.0), AltAzCoord::new(PI/4.0, PI)];
        let stars: Vec<Star> = vec![Star::new(1.0, false), Star::new(0.0, true)];

        let alt_az_list: Vec<Option<AltAzCoord>> = vec![
            Some(AltAzCoord::new(PI/2.0, 0.0))
        ];

        let generator = MagnitudeMapGenerator::new(alt_az_coords, stars);
        let expected: Vec<Option<Average>> = vec![Average::calc_weighted(&vec![0.0, 1.0], &vec![1.0, 1.0])];

        assert_eq!(generator.query_many(alt_az_list, distance), expected);
    }
}