use std::collections::HashMap;

fn as_frequences(data: &[usize]) -> Vec<f32> {
    let mut counter: HashMap<usize, usize> = HashMap::new();
    for &x in data.iter() {
        *counter.entry(x).or_default() += 1;
    }
    counter
        .into_values()
        .map(|x| (x as f32) / (data.len() as f32))
        .collect()
}

pub fn gini_impurity(data: &[usize]) -> f32 {
    let freqs = as_frequences(&data);
    let mut impurity = 0f32;
    for p in freqs.into_iter() {
        impurity += p * (1. - p)
    }

    impurity
}

pub fn entropy(data: &[usize]) -> f32 {
    let freqs = as_frequences(data);
    let mut entropy = 0f32;
    for p in freqs.into_iter() {
        entropy += -(p * p.log2())
    }
    entropy
}

fn rss(data: Vec<f32>) -> f32 {
    let avg: f32 = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|x| (x - avg).powi(2)).sum()
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_gini() {
        let xs = vec![1, 1, 1, 1];
        assert!(gini_impurity(&xs) < 0.1);

        let xs = vec![1, 1, 2, 2];
        assert!(gini_impurity(&xs) < 0.6);
        assert!(gini_impurity(&xs) > 0.4);
    }
}
