mod entry;
mod star_counter;
mod neighbour_list;
mod average;
mod weighted_value;

use pyo3::impl_::wrap::OkWrap;
use pyo3::prelude::*;
use crate::average::Average;

use crate::entry::Entry;
use crate::neighbour_list::NeighbourList;

#[pyfunction]
fn gen_cloud_map(stars: Vec<Entry>, box_size: f64, x_limit: f64, y_limit: f64) -> Vec<Vec<Option<Average>>>
{
    let mut neighbour_list = NeighbourList::new(box_size, x_limit, y_limit);
    for star in stars
    {
        neighbour_list.insert_entry(star);
    }

    neighbour_list.calc_vis_map()
}


#[pymodule]
fn cloudmap_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gen_cloud_map, m)?)?;
    m.add_class::<Entry>()?;
    m.add_class::<Average>()?;
    Ok(())
}
