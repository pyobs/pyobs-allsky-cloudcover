use pyo3::prelude::*;

#[pyclass]
pub struct Average
{
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    std: f64
}

impl Average
{
    pub fn calc(values: &Vec<f64>) -> Option<Average>
    {
        if values.is_empty()
        {
            return None
        }

        let length: f64 = values.len() as f64;

        let sum: f64 = values.iter().fold(0.0, |acc, x| acc + x);
        let average: f64 = sum/length;

        let std: f64 = (values.iter().fold(0.0, |acc, x| acc + (x - average).powf(2.0))/length).sqrt();

        Some(Average {value: average, std})
    }

    pub fn calc_weighted(values: &Vec<f64>, weights: &Vec<f64>) -> Option<Average>
    {
        let average = Self::calc(values)?;

        let norm: f64 = weights.iter().fold(0.0, |acc, x| acc + x);

        let sum: f64 = values.iter().zip(weights).fold(0.0, |acc, (x, w)| acc + x * w);
        let weighted_average: f64 = sum/norm;

        let weighted_std: f64 = average.get_std() * weights.iter().fold(0.0, |acc, x| acc + x.powf(2.0)).sqrt();

        Some(Average {value: weighted_average, std: weighted_std})
    }

    pub fn get_value(&self) -> f64
    {
        self.value
    }

    pub fn get_std(&self) -> f64
    {
        self.std
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_calc_average()
    {
        let values: Vec<f64> = vec![1.0, 3.0];

        let average_result = Average::calc(&values).unwrap();

        assert_eq!(average_result.get_value(), 2.0);
        assert_eq!(average_result.get_std(), 1.0);

    }
}