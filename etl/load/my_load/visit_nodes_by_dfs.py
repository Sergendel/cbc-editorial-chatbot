import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from config.config import Config

logger = logging.getLogger(__name__)


class DFSBase(ABC):

    @abstractmethod
    def visit_nodes(self, dataholder):
        pass


class DataHolder:
    def __init__(self, config):
        self.json_file_path = config.processed_guidelines_path
        self.data = self.load_data(self.json_file_path)

    @staticmethod
    def load_data(file_path):

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Loaded guidelines from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load guidelines: {e}")
            raise


class VisitJsonDFS(DFSBase):

    def visit_nodes(self, dataholder: DataHolder):
        data = dataholder.data

        def dfs(current_data, path=None):
            if path is None:
                path = []

            if isinstance(current_data, dict):
                for key, value in current_data.items():
                    if key == "metadata":
                        print(
                            f"Metadata at "
                            f"{' > '.join(path) if path else 'root'}: {value}"
                        )
                        continue

                    new_path = path + [key]
                    print(f"Visiting node: {' > '.join(new_path)}")

                    dfs(value, new_path)

            elif isinstance(current_data, str):
                node_path = " > ".join(path)
                print(f"Reached leaf node at {node_path}:")
                # print(f"Content: {current_data[:100]}...")  # Short preview

        # Start traversal from root
        dfs(data)


if __name__ == "__main__":

    # load config
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    # load data
    dataholder = DataHolder(config)

    # visit
    dfs_visitor = VisitJsonDFS()
    dfs_visitor.visit_nodes(dataholder)
