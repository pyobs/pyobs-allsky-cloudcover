mod star;
mod star_counter;
mod average;
mod weighted_value;
mod alt_az_coord;
mod cloud_map_generator;
mod day;

use pyo3::prelude::*;
use crate::alt_az_coord::AltAzCoord;
use crate::average::Average;
use crate::cloud_map_generator::MagnitudeMapGenerator;

use crate::star::Star;



#[pymodule]
fn cloudmap_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Star>()?;
    m.add_class::<Average>()?;
    m.add_class::<AltAzCoord>()?;
    m.add_class::<MagnitudeMapGenerator>()?;
    Ok(())
}

