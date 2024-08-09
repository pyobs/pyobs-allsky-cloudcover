use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Star
{
    #[pyo3(get, set)]
    v_mag: f64,
    #[pyo3(get, set)]
    found: bool

}
#[pymethods]
impl Star
{
    #[new]
    pub fn new(v_mag: f64, found: bool) -> Self {
        Star {v_mag, found }
    }
}

impl Star
{
    pub fn get_v_mag(&self) -> f64
    {
        self.v_mag
    }

    pub fn get_found(&self) -> bool
    {
        self.found
    }
}