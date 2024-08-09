mod night;
pub mod sky_query;

use pyo3::prelude::*;
use crate::night::alt_az_coord::AltAzCoord;
use crate::night::average::Average;
use crate::night::cloud_map_generator::MagnitudeMapGenerator;
use crate::night::star::Star;
use crate::sky_query::SkyPixelQuery;

#[pymodule]
fn cloudmap_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Star>()?;
    m.add_class::<Average>()?;
    m.add_class::<AltAzCoord>()?;
    m.add_class::<MagnitudeMapGenerator>()?;
    m.add_class::<SkyPixelQuery>()?;
    Ok(())
}

