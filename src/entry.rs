use pyo3::prelude::*;

#[pyclass]
#[pyo3(text_signature = "(sao, px, py, found, /)")]
#[derive(Clone)]
pub struct Entry
{
    #[pyo3(get, set)]
    SAO: u64,
    #[pyo3(get, set)]
    CAT_PX: f64,
    #[pyo3(get, set)]
    CAT_PY: f64,
    #[pyo3(get, set)]
    Vmag: f64,
    #[pyo3(get, set)]
    found: bool

}
#[pymethods]
impl Entry
{
    #[new]
    pub fn new(sao: u64, px: f64, py: f64, v_mag: f64, found: bool) -> Self {
        Entry {SAO: sao, CAT_PX: px, CAT_PY: py, Vmag: v_mag, found }
    }
}

impl Entry
{
    pub fn get_px(&self) -> f64
    {
        self.CAT_PX
    }

    pub fn get_py(&self) -> f64
    {
        self.CAT_PY
    }

    pub fn get_v_mag(&self) -> f64
    {
        self.Vmag
    }

    pub fn get_found(&self) -> bool
    {
        self.found
    }
}