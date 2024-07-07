use std::f64::consts::PI;
use ball_tree::{BallTree, Query};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::alt_az_coord::AltAzCoord;
use crate::average::Average;
use crate::entry::Entry;
use crate::star_counter::StarCounter;
use crate::weighted_value::WeightedValue;

#[pyclass]
pub struct MagnitudeMapGenerator
{
    neighbours: Vec<Vec<Option<Vec<(usize, f64)>>>>,
    length: usize,
}

#[pymethods]
impl MagnitudeMapGenerator
{
    #[new]
    pub fn new(alt_az_coords: Vec<AltAzCoord>, image_alt_az_coords: Vec<Vec<Option<AltAzCoord>>>, distance: f64) -> MagnitudeMapGenerator
    {
        let length = alt_az_coords.len();
        let indexes: Vec<usize> = (0..length).collect();

        let ball_tree = BallTree::new(alt_az_coords, indexes);

        let neighbours:  Vec<Vec<Option<Vec<(usize, f64)>>>> = image_alt_az_coords.into_par_iter().map(
            |image_row_alt_az|
                image_row_alt_az.into_par_iter().map(
                    |coord| calc_neighbours(&ball_tree, &coord, &distance)
                ).collect()
        ).collect();

        MagnitudeMapGenerator {neighbours, length}
    }

    pub fn gen_cloud_map(&self, stars: Vec<Entry>) -> Vec<Vec<Option<Average>>>
    {
        assert_eq!(stars.len(), self.length, "Stars length must match length of coordinates!");

        let vis_map: Vec<Vec<Option<Average>>> = (&self.neighbours).into_par_iter().map(
            |image_row_alt_az|
                image_row_alt_az.into_par_iter().map(
                    |neighbours| calc_visibility(&stars, neighbours)
                ).collect()
        ).collect();

        vis_map
    }
}

fn calc_neighbours(ball_tree: &BallTree<AltAzCoord, usize>, coord: &Option<AltAzCoord>, distance: &f64) -> Option<Vec<(usize, f64)>>
{
    let mut query = ball_tree.query();
    if let Some(altazcoord) = coord
    {
        let indices = query.nn_within(altazcoord, *distance).map(|(_, distance, index)| (*index, distance)).collect();
        return Some(indices)
    }

    return None
}

fn calc_visibility(stars: &Vec<Entry>, neighbours: &Option<Vec<(usize, f64)>>) -> Option<Average>
{
    let mut star_counter = StarCounter::default();

    if let Some(neighbours) = neighbours
    {
        for (index, distance) in neighbours
        {
            star_counter.increment_stars();

            let entry = &stars[*index];

            let weighted_vmag = WeightedValue::new(entry.get_v_mag(), (-distance * PI/(7.0 * 180.0)).exp()); //*distance
            if entry.get_found()
            {
                star_counter.add_visible_v_mag(weighted_vmag)
            }else {
                star_counter.increment_n_visible();
                star_counter.add_n_visible_v_mag(weighted_vmag)
            }
        }
    }

    star_counter.calc_v_mag_border_value()
}


#[cfg(test)]
mod tests
{
    use std::f64::consts::PI;
    use super::*;


    #[test]
    fn test_gen_cloud_map()
    {
        let alt_az_list: Vec<Vec<Option<AltAzCoord>>> = vec![
            vec![Some(AltAzCoord::new(PI/2.0, 0.0))]
        ];

        let alt_az_coords: Vec<AltAzCoord> = vec![AltAzCoord::new(PI/4.0, 0.0), AltAzCoord::new(PI/4.0, PI)];

        let generator = MagnitudeMapGenerator::new(alt_az_coords, alt_az_list, PI/2.0);

        let stars: Vec<Entry> = vec![Entry::new(1.0, false), Entry::new(0.0, true)];

        let expected: Vec<Vec<Option<Average>>> = vec![vec![Average::calc_weighted(&vec![0.0, 1.0], &vec![1.0, 1.0])]];

        assert_eq!(generator.gen_cloud_map(stars), expected);
    }
}