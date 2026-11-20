from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from blossom.context.context import Context
from blossom.dataframe.aggregate import AggregateFunc
from blossom.dataframe.data_handler import DataHandler
from blossom.op.operator import Operator
from blossom.schema.schema import Schema

if TYPE_CHECKING:
    from blossom.annotation.association import AnnotationJoinConfig
    from blossom.annotation.table import VersionedAnnotationTable


class Dataset(ABC):
    """
    Abstract base class for datasets.

    This class defines the interface for all dataset implementations,
    providing methods for data transformation, filtering, aggregation,
    and I/O operations.
    """

    def __init__(self, context: Optional[Context] = None):
        """
        Initialize a dataset with an optional context.

        Args:
            context: The execution context (optional)
        """
        self.context = context or Context()

    @abstractmethod
    def map(self, func: Callable[[Schema], Schema]) -> "Dataset":
        """
        Apply a function to each item in the dataset.

        Args:
            func: Function that takes a schema and returns a transformed schema

        Returns:
            A new dataset with the transformed items
        """
        pass

    @abstractmethod
    def filter(self, func: Callable[[Schema], bool]) -> "Dataset":
        """
        Filter items in the dataset based on a predicate function.

        Args:
            func: Predicate function that returns True for items to keep

        Returns:
            A new dataset with only the items that satisfy the predicate
        """
        pass

    @abstractmethod
    def transform(self, func: Callable[[list[Schema]], list[Schema]]) -> "Dataset":
        """
        Apply a transformation to batches of items in the dataset.

        Args:
            func: Function that takes a list of schemas and returns a transformed list

        Returns:
            A new dataset with the transformed items
        """
        pass

    @abstractmethod
    def sort(self, func: Callable[[Schema], Any], ascending: bool = True) -> "Dataset":
        """
        Sort the dataset based on a key function.

        Args:
            func: Function that extracts the sort key from each item
            ascending: Whether to sort in ascending order (default: True)

        Returns:
            A new dataset with sorted items
        """
        pass

    @abstractmethod
    def execute(self, operators: list[Operator]) -> "Dataset":
        """
        Execute a sequence of operators on the dataset.

        Args:
            operators: List of operators to apply

        Returns:
            A new dataset with the operators applied
        """
        pass

    @abstractmethod
    def collect(self) -> list[Schema]:
        """
        Collect all items in the dataset into a list.

        Returns:
            List of all items in the dataset
        """
        pass

    @abstractmethod
    def limit(self, num_rows: int) -> "Dataset":
        """
        Limit the dataset to a specified number of items.

        Args:
            num_rows: Maximum number of items to keep

        Returns:
            A new dataset with at most num_rows items
        """
        pass

    @abstractmethod
    def shuffle(self) -> "Dataset":
        """
        Randomly shuffle the items in the dataset.

        Returns:
            A new dataset with shuffled items
        """
        pass

    @abstractmethod
    def repartition(self, num_partitions: int) -> "Dataset":
        """
        Repartition the dataset into a specified number of partitions.

        Args:
            num_partitions: Number of partitions

        Returns:
            A new dataset with the specified number of partitions
        """
        pass

    @abstractmethod
    def split(self, n: int) -> list["Dataset"]:
        """
        Split the dataset into n equal parts.

        Args:
            n: Number of parts to split into

        Returns:
            List of n datasets
        """
        pass

    @abstractmethod
    def aggregate(self, *aggs: AggregateFunc) -> Union[Any, dict[str, Any]]:
        """
        Aggregate the dataset using one or more aggregate functions.

        Args:
            *aggs: Aggregate functions to apply

        Returns:
            Single aggregation result or dictionary of named results
        """
        pass

    @abstractmethod
    def group_by(
        self, func: Callable[[Schema], Any], name: str = "group"
    ) -> "GroupedDataset":
        """
        Group the dataset by a key function.

        Args:
            func: Function that extracts the grouping key from each item
            name: Name of the grouping key in the result (default: "group")

        Returns:
            A grouped dataset
        """
        pass

    @abstractmethod
    def union(self, others: Union["Dataset", list["Dataset"]]) -> "Dataset":
        """
        Union this dataset with one or more other datasets.

        Args:
            others: Dataset or list of datasets to union with

        Returns:
            A new dataset containing all items from all datasets
        """
        pass

    @abstractmethod
    def cache(self) -> "Dataset":
        """
        Cache the dataset in memory.

        Returns:
            The cached dataset
        """
        pass

    @abstractmethod
    def from_list(self, schemas: list[Schema]) -> "Dataset":
        """
        Create a dataset from a list of schemas.

        Args:
            schemas: List of schemas

        Returns:
            A new dataset containing the schemas
        """
        pass

    @abstractmethod
    def read_json(
        self, path: Union[str, list[str]], data_handler: Optional[DataHandler] = None
    ) -> "Dataset":
        """
        Read a dataset from JSON file(s).

        Args:
            path: Path to the JSON file or list of paths to JSON files.
            data_handler: Optional data handler for custom deserialization

        Returns:
            A new dataset containing the items from the JSON file(s)
        """
        pass

    @abstractmethod
    def write_json(self, path: str, data_handler: Optional[DataHandler] = None) -> None:
        """
        Write the dataset to a JSON file.

        Args:
            path: Path to the JSON file
            data_handler: Optional data handler for custom serialization
        """
        pass

    @abstractmethod
    def read_parquet(
        self, path: Union[str, list[str]], data_handler: Optional[DataHandler] = None
    ) -> "Dataset":
        """
        Read a dataset from Parquet file(s).

        Args:
            path: Path to the Parquet file or list of paths to Parquet files.
            data_handler: Optional data handler for custom deserialization

        Returns:
            A new dataset containing the items from the Parquet file(s)
        """
        pass

    @abstractmethod
    def write_parquet(
        self, path: str, data_handler: Optional[DataHandler] = None
    ) -> None:
        """
        Write the dataset to a Parquet file.

        Args:
            path: Path to the Parquet file
            data_handler: Optional data handler for custom serialization
        """
        pass


    @abstractmethod
    def add_metadata(self, func: Callable[[Schema], dict[str, Any]]) -> "Dataset":
        """
        Add metadata to each item in the dataset.

        Args:
            func: Function that takes a schema and returns metadata to add

        Returns:
            A new dataset with added metadata
        """
        pass

    @abstractmethod
    def drop_metadata(self, keys: list[str]) -> "Dataset":
        """
        Drop specified metadata keys from each item in the dataset.

        Args:
            keys: List of metadata keys to drop

        Returns:
            A new dataset with the specified metadata keys dropped
        """
        pass

    def apply_annotations(
        self,
        table: "VersionedAnnotationTable",
        config: Optional["AnnotationJoinConfig"] = None,
    ) -> "Dataset":
        """
        Join the dataset with a versioned external annotation release.

        This is not a metadata-only merge: the newest in-release revision of
        every sample replaces its label and training weight, and samples whose
        newest revision rejects or revokes the annotation (or is still pending
        when pending records are excluded) are removed from the dataset.

        Args:
            table: Versioned table holding the full annotation revision
                history.
            config: Optional join configuration (release selection, orphan and
                unannotated policies, weight defaults, ...).

        Returns:
            A new dataset whose labels, weights and population reflect the
            selected annotation release. Each retained item records the exact
            release/revision applied in its metadata, which survives
            JSON/Parquet write and re-read.
        """
        from blossom.annotation import AnnotationJoinConfig
        from blossom.op.common.annotation_join_operator import (
            AnnotationJoinOperator,
        )

        return self.execute(
            [AnnotationJoinOperator(table, config or AnnotationJoinConfig())]
        )

    def annotation_releases(self) -> list[str]:
        """
        Return the annotation releases currently proven on this dataset.

        Reads the annotation provenance written by :meth:`apply_annotations`,
        including after a JSON/Parquet round trip, so the annotation version
        behind stored training data is always recoverable.
        """
        from blossom.annotation import collect_annotation_releases

        return collect_annotation_releases(self.collect())

    @abstractmethod
    def sum(self, func: Callable[[Schema], Union[int, float]]) -> Union[int, float]:
        """
        Calculate the sum of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Sum of the extracted values
        """
        pass

    @abstractmethod
    def mean(self, func: Callable[[Schema], Union[int, float]]) -> Union[int, float]:
        """
        Calculate the mean of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Mean of the extracted values
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Count the number of items in the dataset.

        Returns:
            Number of items
        """
        pass

    @abstractmethod
    def min(self, func: Callable[[Schema], Union[int, float]]) -> Union[int, float]:
        """
        Find the minimum value of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Minimum of the extracted values
        """
        pass

    @abstractmethod
    def max(self, func: Callable[[Schema], Union[int, float]]) -> Union[int, float]:
        """
        Find the maximum value of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Maximum of the extracted values
        """
        pass

    @abstractmethod
    def variance(
        self, func: Callable[[Schema], Union[int, float]]
    ) -> Union[int, float]:
        """
        Calculate the variance of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Variance of the extracted values
        """
        pass

    @abstractmethod
    def stddev(self, func: Callable[[Schema], Union[int, float]]) -> Union[int, float]:
        """
        Calculate the standard deviation of a numeric function applied to each item.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            Standard deviation of the extracted values
        """
        pass

    @abstractmethod
    def unique(self, func: Callable[[Schema], set[Any]]) -> list[Any]:
        """
        Find unique values of a function applied to each item.

        Args:
            func: Function that extracts values from each item

        Returns:
            List of unique values
        """
        pass


class GroupedDataset(ABC):
    """
    Abstract base class for grouped datasets.

    This class defines the interface for grouped dataset operations,
    providing methods for aggregation and other group-wise operations.
    """

    def __init__(self, context: Context):
        """
        Initialize a grouped dataset with a context.

        Args:
            context: The execution context
        """
        self.context = context

    @abstractmethod
    def aggregate(self, *aggs: AggregateFunc) -> "Dataset":
        """
        Aggregate the grouped dataset using one or more aggregate functions.

        Args:
            *aggs: Aggregate functions to apply to each group

        Returns:
            A dataset with one row per group containing the aggregation results
        """
        pass

    @abstractmethod
    def sum(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Calculate the sum of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the sum
        """
        pass

    @abstractmethod
    def mean(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Calculate the mean of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the mean
        """
        pass

    @abstractmethod
    def count(self) -> "Dataset":
        """
        Count the number of items in each group.

        Returns:
            A dataset with one row per group containing the count
        """
        pass

    @abstractmethod
    def min(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Find the minimum value of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the minimum value
        """
        pass

    @abstractmethod
    def max(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Find the maximum value of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the maximum value
        """
        pass

    @abstractmethod
    def variance(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Calculate the variance of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the variance
        """
        pass

    @abstractmethod
    def stddev(self, func: Callable[[Schema], Union[int, float]]) -> "Dataset":
        """
        Calculate the standard deviation of a numeric function for each group.

        Args:
            func: Function that extracts a numeric value from each item

        Returns:
            A dataset with one row per group containing the standard deviation
        """
        pass

    @abstractmethod
    def unique(self, func: Callable[[Schema], set[Any]]) -> "Dataset":
        """
        Find unique values of a function for each group.

        Args:
            func: Function that extracts values from each item

        Returns:
            A dataset with one row per group containing the unique values
        """
        pass
