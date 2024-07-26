
#[derive(PartialEq, PartialOrd, Debug)]
pub struct WeightedValue
{
    value: f64,
    weight: f64
}

impl WeightedValue
{
    pub fn new(value: f64, weight: f64) -> WeightedValue
    {
        WeightedValue {value, weight}
    }

    pub fn get_value(&self)-> f64
    {
        self.value
    }

    pub fn get_weight(&self) -> f64
    {
        self.weight
    }
}