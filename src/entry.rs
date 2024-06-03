use pyo3::prelude::*;

#[pyclass]
#[pyo3(text_signature = "(sao, px, py, found, /)")]
#[derive(Clone)]
pub struct Entry
{
    #[pyo3(get, set)]
    v_mag: f64,
    #[pyo3(get, set)]
    found: bool

}
#[pymethods]
impl Entry
{
    #[new]
    pub fn new(v_mag: f64, found: bool) -> Self {
        Entry {v_mag, found }
    }
}

impl Entry
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