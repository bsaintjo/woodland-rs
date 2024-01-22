use petgraph::graph::NodeIndex;
use petgraph::Graph;

use crate::criterion;

struct Data {
    targets: Vec<usize>,
    features: Vec<Vec<f32>>,
}

struct DTreeBuilder;

impl DTreeBuilder {
    fn fit(&self, data: &Data) -> DecisionTree {
        let tree = Graph::default();
        while self.should_stop(&tree) {

        }
        todo!()
    }

    fn should_stop(&self, tree: &Graph<DTNode, ()>) -> bool {
        todo!()
    }

}

struct DecisionTree {
    tree: Graph<DTNode, ()>,
    root: NodeIndex,
}

impl DecisionTree {
    fn new() -> Self {
        let mut tree = Graph::default();
        let root = tree.add_node(todo!());
        DecisionTree { tree, root }
    }

    fn fit(data: &Data) {
        todo!()
    }

    fn predict(features: Vec<Vec<f32>>) -> usize {
        todo!()
    }

    fn split_node(&mut self, data: &Data) {
        if let Some(s) = split_data(data) {
            self.tree.add_node(todo!());
            self.tree.add_node(todo!());
        }
    }
}

fn split_data(data: &Data) -> Option<(usize, f32)> {
    let total_info = criterion::gini_impurity(&data.targets);
    let mut best_gain = 0.0f32;
    let mut best_criterion = None;

    // f32 doesn't implement Eq so we can't put them in
    // a HashSet, so we may have duplicates

    for (feat_idx, column) in data.features.iter().enumerate() {
        for value in column {
            let less_than: Vec<_> = column
                .iter()
                .enumerate()
                .filter(|x| x.1 <= value)
                .map(|x| x.0)
                .collect();
            let greater_than: Vec<_> = column
                .iter()
                .enumerate()
                .filter(|x| x.1 > value)
                .map(|x| x.0)
                .collect();
            let lt_targets: Vec<_> = less_than.iter().map(|&idx| data.targets[idx]).collect();
            let gt_targets: Vec<_> = greater_than.iter().map(|&idx| data.targets[idx]).collect();

            let lt_info = (lt_targets.len() as f32) * criterion::gini_impurity(&lt_targets);
            let gt_info = (gt_targets.len() as f32) * criterion::gini_impurity(&lt_targets);

            let gain = total_info - (lt_info + gt_info);
            if gain > best_gain {
                best_gain = gain;
                best_criterion = Some((feat_idx, *value));
            }
        }
    }
    best_criterion
}

struct DTNode {
    criterion: f32,
    samples: usize,
    depth: usize,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_decision_tree() {
        let tree = DecisionTree::new();
    }
}
