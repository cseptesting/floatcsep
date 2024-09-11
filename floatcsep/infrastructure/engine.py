from collections import OrderedDict
from typing import Union, Any


class Task:
    """
    Represents a unit of work to be executed later as part of a task graph.

    A Task wraps an object instance, a method, and its arguments to allow for deferred
    execution. This is useful in workflows where tasks need to be executed in a specific order,
    often dictated by dependencies on other tasks.

    For instance, can wrap a floatcsep.model.Model, its method 'create_forecast' and the
    argument 'time_window', which can be executed later with Task.call() when, for example,
    task dependencies (parent nodes) have been completed.

    Args:
            instance (object): The instance whose method will be executed later.
            method (str): The method of the instance that will be called.
            **kwargs: Arguments to pass to the method when it is invoked.

    """

    def __init__(self, instance: object, method: str, **kwargs):

        self.obj = instance
        self.method = method
        self.kwargs = kwargs

        self.store = None  # Bool for nested tasks.

    def sign_match(self, obj: Union[object, str] = None, meth: str = None, kw_arg: Any = None):
        """
        Checks whether the task matches a given function signature.

        This method is used to verify if a task belongs to a given object, method, or if it
        uses a specific keyword argument. Useful for identifying tasks in a graph based on
        partial matches of their attributes.

        Args:
            obj: The object instance or its name (str) to match against.
            meth: The method name to match against.
            kw_arg: A specific keyword argument value to match against in the task's arguments.

        Returns:
            bool: True if the task matches the provided signature, False otherwise.
        """

        if self.obj == obj or obj == getattr(self.obj, "name", None):
            if meth == self.method:
                if kw_arg in self.kwargs.values():
                    return True
        return False

    def __str__(self):
        """
        Returns a string representation of the task, including the instance name, method, and
        arguments. Useful for debugging purposes.

        Returns:
            str: A formatted string describing the task.
        """
        task_str = f"{self.__class__}\n\t" f"Instance: {self.obj.__class__.__name__}\n"
        a = getattr(self.obj, "name", None)
        if a:
            task_str += f"\tName: {a}\n"
        task_str += f"\tMethod: {self.method}\n"
        for i, j in self.kwargs.items():
            task_str += f"\t\t{i}: {j} \n"

        return task_str[:-2]

    def run(self):
        """
        Executes the task by calling the method on the object instance with the stored
        arguments. If the instance has a `store` attribute, it will use that instead of the
        instance itself. Once executed, the result is stored in the `store` attribute if any
        output is produced.

        Returns:
            The output of the method execution, or None if the method does not return anything.
        """

        if hasattr(self.obj, "store"):
            self.obj = self.obj.store
        output = getattr(self.obj, self.method)(**self.kwargs)

        if output:
            self.store = output
            del self.obj

        return output

    def __call__(self, *args, **kwargs):
        """
        A callable alias for the `run` method. Allows the task to be invoked directly.

        Returns:
            The result of the `run` method.
        """
        return self.run()


class TaskGraph:
    """
    Context manager of floatcsep workload distribution.

    A TaskGraph is responsible for adding tasks, managing dependencies between tasks, and
    executing  tasks in the correct order. Tasks in the graph can depend on one another, and
    the graph ensures that each task is run after all of its dependencies have been satisfied.
    Contains a `Task` dictionary whose dict_keys are the Task to be executed with dict_values
    as the Task's dependencies.

    """

    def __init__(self) -> None:
        """
        Initializes the TaskGraph with an empty task dictionary and task count.
        """
        self.tasks = OrderedDict()
        self._ntasks = 0
        self.name = "floatcsep.infrastructure.engine.TaskGraph"

    @property
    def ntasks(self) -> int:
        """
        Returns the number of tasks currently in the graph.

        Returns:
            int: The total number of tasks in the graph.
        """
        return self._ntasks

    @ntasks.setter
    def ntasks(self, n):
        self._ntasks = n

    def add(self, task: Task):
        """
        Adds a new task to the task graph.

        The task is added to the dictionary of tasks with no dependencies by default.

        Args:
            task (Task): The task to be added to the graph.
        """
        self.tasks[task] = []
        self.ntasks += 1

    def add_dependency(self, task, dep_inst: Union[object, str] = None, dep_meth: str = None,
                       dkw: Any = None):
        """
        Adds a dependency to a task already within the graph.

        Searches for other tasks within the graph whose signature matches the provided
        object instance, method name, or keyword argument. Any matches are added as
        dependencies to the provided task.

        Args:
            task (Task): The task to which dependencies will be added.
            dep_inst: The object instance or name of the dependency.
            dep_meth: The method name of the dependency.
            dkw: A specific keyword argument value of the dependency.

        Returns:
            None
        """
        deps = []
        for i, other_tasks in enumerate(self.tasks.keys()):
            if other_tasks.sign_match(dep_inst, dep_meth, dkw):
                deps.append(other_tasks)

        self.tasks[task].extend(deps)

    def run(self):
        """
        Executes all tasks in the task graph in the correct order based on dependencies.

        Iterates over each task in the graph and runs it after its dependencies have been
        resolved.

        Returns:
            None
        """
        for task, deps in self.tasks.items():
            task.run()

    def __call__(self, *args, **kwargs):
        """
        A callable alias for the `run` method. Allows the task graph to be invoked directly.

        Returns:
            None
        """
        return self.run()
