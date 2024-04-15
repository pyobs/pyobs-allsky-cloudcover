use crate::average::Average;
use crate::weighted_value::WeightedValue;

#[derive(Default)]
pub struct StarCounter
{
    stars: usize,
    n_visible: usize,
    visible_v_mags: Vec<WeightedValue>,
    n_visible_v_mags: Vec<WeightedValue>
}

impl StarCounter
{
    pub fn increment_stars(&mut self)
    {
        self.stars += 1;
    }

    pub fn increment_n_visible(&mut self)
    {
        self.n_visible += 1;
    }

    pub fn add(&mut self, mut other: StarCounter)
    {
        self.stars += other.stars;
        self.n_visible += other.n_visible;

        self.visible_v_mags.append(&mut other.visible_v_mags);
        self.n_visible_v_mags.append(&mut other.n_visible_v_mags);
    }

    pub fn calc_ratio(&self) -> f64
    {
        if self.stars == 0
        {
            return -1.0;
        }
        (self.n_visible as f64)/(self.stars as f64)
    }

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
        let stars = 4;
        let n_visible = 2;
        let visible_v_mags: Vec<f64> = vec![3.0, 2.0];
        let n_visible_v_mags: Vec<f64> = vec![4.0, 5.0];
        let mut counter = StarCounter {stars, n_visible, visible_v_mags, n_visible_v_mags};
        let border_values = counter.calc_v_mag_border_values().unwrap();

        assert_eq!(border_values[0].to_owned(), 3.0);
        assert_eq!(border_values[1].to_owned(), 4.0);
    }

    #[test]
    fn test_calc_v_mag_border_values_overlap()
    {
        let stars = 4;
        let n_visible = 2;
        let visible_v_mags: Vec<f64> = vec![4.0, 2.0];
        let n_visible_v_mags: Vec<f64> = vec![3.0, 5.0];
        let mut counter = StarCounter {stars, n_visible, visible_v_mags, n_visible_v_mags};
        let border_values = counter.calc_v_mag_border_values().unwrap();

        assert_eq!(border_values[1].to_owned(), 3.0);
        assert_eq!(border_values[0].to_owned(), 4.0);
    }
}