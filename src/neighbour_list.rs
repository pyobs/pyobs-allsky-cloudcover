use rayon::prelude::*;
use crate::average::Average;
use crate::entry::Entry;
use crate::star_counter::StarCounter;
use crate::weighted_value::WeightedValue;

pub struct NeighbourList
{
    box_size: f64,
    x_limit: f64,
    y_limit: f64,
    boxes: Vec<Vec<Vec<Entry>>>
}

impl NeighbourList
{
    pub fn new(box_size: f64, x_limit: f64, y_limit: f64) -> NeighbourList
    {
        let boxes: Vec<Vec<Vec<Entry>>> = vec![vec![vec![]; (x_limit/box_size) as usize + 1]; (y_limit/box_size) as usize + 1];
        NeighbourList{box_size, x_limit, y_limit, boxes}
    }

    fn calc_y_index(&self, py: f64) -> usize
    {
        (py/self.box_size) as usize
    }

    fn calc_x_index(&self, px: f64) -> usize
    {
        (px/self.box_size) as usize
    }

    pub fn insert_entry(&mut self, entry: Entry)
    {
        let ix = self.calc_x_index(entry.get_px());
        let iy = self.calc_y_index(entry.get_py());

        self.boxes[iy][ix].push(entry);
    }

    fn calc_weight(&self, distance: f64) -> f64
    {
        let std = self.box_size.powf(2.0)/(2.0 * (2.0f64.ln() + 5.0f64.ln()));

        (-distance.powf(2.0)/std).exp()
    }

    fn count_neighbor_stars(&self, px: f64, py: f64, ix: usize, iy: usize) -> StarCounter
    {
        let mut counter = StarCounter::default();

        for star in &self.boxes[iy][ix]
        {
            let distance = ((star.get_px() - px).powf(2.0) + (star.get_py() - py).powf(2.0)).sqrt();

            if distance > self.box_size
            {
                continue
            }

            let weight = self.calc_weight(distance);

            let weighted_value = WeightedValue::new(star.get_v_mag(), weight);

            counter.increment_stars();

            if !star.get_found()
            {
                counter.increment_n_visible();
                counter.add_n_visible_v_mag(weighted_value);
            }else {
                counter.add_visible_v_mag(weighted_value);
            }
        }

        counter
    }

    fn calc_av_visibility(&self, px: f64, py: f64) -> Option<Average>
    {
        let ix = self.calc_x_index(px) as i32;
        let iy = self.calc_x_index(py) as i32;

        let dir_x = [0, 1, -1, 0, 0, 1, -1, -1, 1];
        let dir_y = [0, 0, 0, 1, -1, 1, -1, 1, -1];

        let mut counter: StarCounter = StarCounter::default();

        for i in 0..9
        {
            if (ix + dir_x[i]) < 0 || (ix + dir_x[i]) == (self.boxes[0].len() as i32)
            {
                continue
            }else if (iy + dir_y[i]) < 0 || (iy + dir_y[i]) == (self.boxes.len() as i32)
            {
                continue
            }

            counter.add(self.count_neighbor_stars(px, py, (ix + dir_x[i]) as usize, (iy + dir_y[i]) as usize))

        }

        //counter.calc_ratio()
        counter.calc_v_mag_border_value()
    }

    pub fn calc_vis_map(&self) -> Vec<Vec<Option<Average>>>
    {
        let vis_map: Vec<Vec<Option<Average>>> = (0..(self.y_limit as usize)).into_par_iter().map(
            |py|
            (0..(self.x_limit as usize)).into_par_iter().map(
                |px| self.calc_av_visibility(px as f64, py as f64)
            ).collect()
        ).collect();

        vis_map
    }
}