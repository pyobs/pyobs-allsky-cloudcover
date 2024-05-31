mod entry;
mod star_counter;
mod average;
mod weighted_value;
mod alt_az_coord;
mod cloud_map_generator;

use pyo3::impl_::wrap::OkWrap;
use pyo3::prelude::*;
use crate::alt_az_coord::AltAzCoord;
use crate::average::Average;
use crate::cloud_map_generator::CloudMapGenerator;

use crate::entry::Entry;



#[pymodule]
fn allsky_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Entry>()?;
    m.add_class::<Average>()?;
    m.add_class::<AltAzCoord>()?;
    m.add_class::<CloudMapGenerator>()?;
    Ok(())
}

