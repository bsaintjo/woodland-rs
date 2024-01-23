use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;
use petgraph::{data::Build, Graph};

use crate::criterion;

pub struct Data {
    n_cols: usize,
    n_samples: usize,
    targets: Vec<usize>,
    features: Vec<Vec<f32>>,
}

struct SplitData {
    less_than_eq: Data,
    greater_than: Data,
}

type DTree = Graph<TreeNode, bool>;

impl Data {
    fn is_pure(&self) -> bool {
        self.targets.iter().collect::<HashSet<_>>().len() == 1
    }

    fn most_common_target(&self) -> Option<usize> {
        let mut counter: HashMap<&usize, usize> = HashMap::new();
        for target in self.targets.iter() {
            *counter.entry(target).or_default() += 1;
        }
        counter.into_iter().max_by_key(|(_, c)| *c).map(|x| *x.0)
    }

    fn split_data(&self, col_idx: usize, value: f32) -> SplitData {
        let lt_idxs = self.argwhere(col_idx, |x| x <= value);
        let gt_idxs = self.argwhere(col_idx, |x| x > value);

        let lt_data = self.idx(lt_idxs.iter().cloned());
        let gt_data = self.idx(gt_idxs.iter().cloned());
        SplitData {
            less_than_eq: lt_data,
            greater_than: gt_data,
        }
    }

    /// Returns indices where the predicate is true over a column with the Data
    fn argwhere(&self, col_idx: usize, predicate: impl Fn(f32) -> bool) -> HashSet<usize> {
        let column = &self.features[col_idx];
        let idxs: HashSet<_> = column
            .iter()
            .enumerate()
            .filter(|x| predicate(*x.1))
            .map(|x| x.0)
            .collect();
        idxs
    }

    fn split_argwhere(
        &self,
        col_idx: usize,
        predicate: impl Fn(f32) -> bool,
    ) -> (HashSet<usize>, HashSet<usize>) {
        todo!()
    }

    /// Returns a subset of the Data given a set of indices
    fn idx(&self, idxs: impl Iterator<Item = usize>) -> Self {
        let mut targets = Vec::new();
        let mut features = Vec::new();
        for idx in idxs {
            targets.push(self.targets[idx]);
            features.push(self.features[idx].clone());
        }
        Data {
            n_cols: self.n_cols,
            n_samples: self.n_samples,
            targets,
            features,
        }
    }
}

pub struct DecisionTreeBuilder {
    max_depth: usize,
    min_samples: usize,
}

impl DecisionTreeBuilder {
    pub fn new() -> Self {
        DecisionTreeBuilder {
            max_depth: 1,
            min_samples: 2,
        }
    }
    pub fn fit(&self, data: &Data) -> DecisionTree {
        let mut tree = Graph::default();
        let root = TreeNode::from_data(data, &mut tree);
        DecisionTree { tree, root }
    }

    fn should_stop(&self, tree: &DTree) -> bool {
        todo!()
    }
}

pub struct DecisionTree {
    tree: DTree,
    root: NodeIndex,
}

impl DecisionTree {
    fn new() -> Self {
        let mut tree = Graph::default();
        let root = tree.add_node(todo!());
        DecisionTree { tree, root }
    }

    pub fn predict(&self, features: Vec<Vec<f32>>) -> usize {
        todo!()
    }
}

fn best_split(data: &Data) -> Option<(usize, f32)> {
    let total_info = criterion::gini_impurity(&data.targets);
    let mut best_gain = 0.0f32;
    let mut best_criterion = None;

    // f32 doesn't implement Eq so we can't put them in
    // a HashSet, so we may have duplicates

    for (feat_idx, column) in data.features.iter().enumerate() {
        for &value in column {
            let split_data = data.split_data(feat_idx, value);
            let lt_info = (split_data.less_than_eq.targets.len() as f32)
                * criterion::gini_impurity(&split_data.less_than_eq.targets);
            let gt_info = (split_data.greater_than.targets.len() as f32)
                * criterion::gini_impurity(&split_data.greater_than.targets);

            let gain = total_info - (lt_info + gt_info);
            if gain > best_gain {
                best_gain = gain;
                best_criterion = Some((feat_idx, value));
            }
        }
    }
    best_criterion
}

#[derive(Clone)]
enum TreeNode {
    Internal(InternalNode),
    Leaf { outcome: usize },
}

impl TreeNode {
    fn from_data(data: &Data, tree: &mut DTree) -> NodeIndex {
        let total_info = criterion::gini_impurity(&data.targets);
        let mut best_gain = 0.0f32;
        let mut best_criterion = None;
        let mut best_data_split = None;

        // f32 doesn't implement Eq so we can't put them in
        // a HashSet, so we may have duplicates

        for (feat_idx, column) in data.features.iter().enumerate() {
            for &value in column {
                let split_data = data.split_data(feat_idx, value);
                let lt_info = (split_data.less_than_eq.targets.len() as f32)
                    * criterion::gini_impurity(&split_data.less_than_eq.targets);
                let gt_info = (split_data.greater_than.targets.len() as f32)
                    * criterion::gini_impurity(&split_data.greater_than.targets);

                let gain = total_info - (lt_info + gt_info);
                if gain > best_gain {
                    best_gain = gain;
                    best_criterion = Some((feat_idx, value));
                    best_data_split = Some(split_data)
                }
            }
        }
        if let Some((col_idx, value)) = best_criterion {
            let node = Self::Internal(InternalNode {
                criterion: value,
                col_idx,
            });
            let node_idx = tree.add_node(node);
            let lt_idx = TreeNode::from_data(&best_data_split.as_ref().unwrap().less_than_eq, tree);
            let gt_idx = TreeNode::from_data(&best_data_split.unwrap().greater_than, tree);
            tree.add_edge(node_idx, lt_idx, true);
            tree.add_edge(node_idx, gt_idx, true);

            node_idx
        } else {
            let node = Self::Leaf {
                outcome: data.most_common_target().unwrap(),
            };
            tree.add_node(node)
        }
    }
}

#[derive(Clone)]
struct InternalNode {
    criterion: f32,
    col_idx: usize,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_decision_tree() {
        let targets = vec![0, 0, 1, 1];
        let features = vec![
            vec![0.1, 0.9],
            vec![0.1, 0.9],
            vec![20., 0.9],
            vec![20., 0.9],
        ];
        let data = Data {
            n_cols: features[0].len(),
            n_samples: features.len(),
            targets,
            features,
        };
        let tree = DecisionTreeBuilder::new().fit(&data);
        let new_data = vec![vec![20., 0.9f32]];
        let result = tree.predict(new_data);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_data() {
        let mut data = Data {
            n_cols: 1,
            n_samples: 1,
            targets: vec![0],
            features: vec![vec![0.]],
        };
        assert!(data.is_pure());

        data.targets = vec![0, 1];
        assert!(!data.is_pure());
    }
}
