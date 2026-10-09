import json
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """converts tree to json
    """
    def tree_traverse(sub_tree, node_index):
        """travers a tree
        """
        if tree.tree_.feature[node_index] != -2:
            sub_tree['feature_index'] = int(tree.tree_.feature[node_index])
            sub_tree['threshold'] = round(float(tree.tree_.threshold[node_index]), 4)
        
        node_left = tree.tree_.children_left[node_index]
        if node_left != -1:
            sub_tree['left'] = tree_traverse({}, node_left)
        else:
            sub_tree['class'] = int(np.argmax(tree.tree_.value[node_index][0]))
        node_right = tree.tree_.children_right[node_index]
        if node_right != -1:
            sub_tree['right'] = tree_traverse({}, node_right)
        else:
            sub_tree['class'] = int(np.argmax(tree.tree_.value[node_index][0]))
        return sub_tree

    tree = tree_traverse({}, 0)
    return json.dumps(tree)


def generate_sql_query(tree_as_json: str, features: list) -> str:
    """converts json tree to SQL query
    """
    def tree_traversal(sql_query, tree, level, features=features):
        """travers a tree
        """
        sql_query += ' ' * 2 * level + 'CASE\n'
        line = (' ' * 2 * (level + 1) + 'WHEN ' + str(features[tree['feature_index']]) + 
                ' > ' + str(tree['threshold']) + ' THEN')
        if 'class' in tree['right']:
            line += ' ' + str(tree['right']['class']) + '\n'
            sql_query += line
        else:
            sql_query += line + '\n'
            sql_query = tree_traversal(sql_query, tree['right'], level + 2)
        line = ' ' * 2 * (level + 1) + 'ELSE'
        if 'class' in tree['left']:
            sql_query += line + ' ' +  str(tree['left']['class']) + '\n'
        else:
            sql_query += line + '\n'
            sql_query = tree_traversal(sql_query, tree['left'], level + 2)
        sql_query += ' ' * 2 * level
        if level == 0:
            sql_query += 'END AS class_label'
        else:   
            sql_query += 'END\n'
        return sql_query
    tree = json.loads(tree_as_json)
    sql_query = 'SELECT\n'
    level = 0
    sql_query = tree_traversal(sql_query, tree, level)
    return sql_query
