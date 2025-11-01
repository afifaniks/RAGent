import ast
from typing import Union


class SignatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.tree = []

    def visit_Module(self, node):
        for item in node.body:
            self.tree.append(self._process_node(item))

    def _process_node(self, node) -> Union[str, tuple]:
        if isinstance(node, ast.FunctionDef):
            return self._format_function(node)

        elif isinstance(node, ast.ClassDef):
            class_name = f"class {node.name}"
            children = []
            for item in node.body:
                children.append(self._process_node(item))
            return (class_name, children)

    def _format_function(self, node: ast.FunctionDef) -> str:
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        arg_list = ", ".join(args)
        return f"def {node.name}({arg_list})"


def str_signature_tree(tree, indent=0):
    signature_tree = ""
    for node in tree:
        if isinstance(node, str):
            signature_tree += "  " * indent + node + "\n"
        elif isinstance(node, tuple):
            class_name, children = node
            signature_tree += "  " * indent + class_name + "\n"
            signature_tree += str_signature_tree(children, indent + 1)

    return signature_tree


def extract_signature_tree(source: str) -> str:
    tree = ast.parse(source)
    builder = SignatureExtractor()
    builder.visit(tree)
    return str_signature_tree(builder.tree)
