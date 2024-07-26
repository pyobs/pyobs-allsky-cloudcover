use crate::night::average::Average;
use crate::night::weighted_value::WeightedValue;

#[derive(Default)]
pub struct StarCounter
{
    visible_v_mags: Vec<WeightedValue>,
    n_visible_v_mags: Vec<WeightedValue>
}

impl StarCounter
{
    pub fn add_visible_v_mag(&mut self, v_mag: WeightedValue)
    {
        self.visible_v_mags.push(v_mag);
    }

    pub fn add_n_visible_v_mag(&mut self, v_mag: WeightedValue)
    {
        self.n_visible_v_mags.push(v_mag);
    }

    fn calc_v_mag_border_values(&self) -> Option<Vec<&WeightedValue>>
    {
        let highest_visible: &WeightedValue;
        let lowest_n_visible: &WeightedValue;
        if self.visible_v_mags.len() == 0 && self.n_visible_v_mags.len() == 0
        {
            return None
        } else if self.visible_v_mags.len() == 0 {
            lowest_n_visible = self.n_visible_v_mags.iter().min_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap()).unwrap();
            highest_visible = lowest_n_visible;
        }else if self.n_visible_v_mags.len() == 0 {
            highest_visible = self.visible_v_mags.iter().max_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap()).unwrap();
            lowest_n_visible = highest_visible;
        } else {
            highest_visible = self.visible_v_mags.iter().max_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap()).unwrap();
            lowest_n_visible = self.n_visible_v_mags.iter().min_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap()).unwrap();
        }

        if highest_visible > lowest_n_visible
        {
            let vis_border= self.visible_v_mags.iter().filter(|a| *lowest_n_visible <= **a && **a <= *highest_visible);
            let n_vis_border = self.n_visible_v_mags.iter().filter(|a| *lowest_n_visible <= **a && **a <= *highest_visible);

            return Some(vis_border.chain(n_vis_border).collect())
        }

        return Some(vec![highest_visible, lowest_n_visible]);
    }

    pub fn calc_v_mag_border_value(&self) -> Option<Average>
    {
        let border_values = self.calc_v_mag_border_values()?;
        let values: Vec<f64> = border_values.iter().map(|x| x.get_value()).collect();
        let weights: Vec<f64> = border_values.iter().map(|x| x.get_weight()).collect();
        Average::calc_weighted(&values, &weights)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_calc_v_mag_border_values_no_overlap()
    {
        let visible_v_mags: Vec<WeightedValue> = vec![WeightedValue::new(3.0, 1.0), WeightedValue::new(3.0, 2.0)];
        let n_visible_v_mags: Vec<WeightedValue> = vec![WeightedValue::new(4.0, 1.0), WeightedValue::new(5.0, 1.0)];
        let counter = StarCounter {visible_v_mags, n_visible_v_mags};
        let border_values = counter.calc_v_mag_border_values().unwrap();

        assert_eq!(border_values[0].to_owned().get_value(), 3.0);
        assert_eq!(border_values[1].to_owned().get_value(), 4.0);
    }

    #[test]
    fn test_calc_v_mag_border_values_overlap()
    {
        let visible_v_mags: Vec<WeightedValue> = vec![WeightedValue::new(4.0, 1.0), WeightedValue::new(3.0, 2.0)];
        let n_visible_v_mags: Vec<WeightedValue> = vec![WeightedValue::new(3.0, 1.0), WeightedValue::new(5.0, 1.0)];
        let counter = StarCounter {visible_v_mags, n_visible_v_mags};
        let border_values = counter.calc_v_mag_border_values().unwrap();

        assert_eq!(border_values[1].to_owned().get_value(), 3.0);
        assert_eq!(border_values[0].to_owned().get_value(), 4.0);
    }
}