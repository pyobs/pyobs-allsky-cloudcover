use ball_tree::{BallTree, Query};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::alt_az_coord::AltAzCoord;
use crate::average::Average;
use crate::entry::Entry;
use crate::star_counter::StarCounter;
use crate::weighted_value::WeightedValue;

#[pyclass]
pub struct CloudMapGenerator
{
    neighbours: Vec<Vec<Option<Vec<(usize, f64)>>>>,
    length: usize,
}

#[pymethods]
impl CloudMapGenerator
{
    #[new]
    pub fn new(alt_az_coords: Vec<AltAzCoord>, image_alt_az_coords: Vec<Vec<Option<AltAzCoord>>>, distance: f64) -> CloudMapGenerator
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

        CloudMapGenerator {neighbours, length}
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

            let weighted_vmag = WeightedValue::new(entry.get_v_mag(), *distance);
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