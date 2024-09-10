from collections import OrderedDict


class Task:

    def __init__(self, instance, method, **kwargs):
        """
        Base node of the workload distribution. Wraps lazily objects, methods and their
        arguments for them to be executed later. For instance, can wrap a floatcsep.Model, its
        method 'create_forecast' and the argument 'time_window', which can be executed later
        with Task.call() when, for example, task dependencies (parent nodes) have been completed.

        Args:
            instance: can be floatcsep.Experiment, floatcsep.Model, floatcsep.Evaluation
            method: the instance's method to be lazily created
            **kwargs: keyword arguments passed to method.
        """

        self.obj = instance
        self.method = method
        self.kwargs = kwargs

        self.store = None  # Bool for nested tasks. DEPRECATED

    def sign_match(self, obj=None, met=None, kw_arg=None):
        """
        Checks if the Task matches a given signature for simplicity.

        Purpose is to check from the outside if the Task is from a given object
        (Model, Experiment, etc.), matching either name or object or description
        Args:
            obj: Instance or instance's name str. Instance is preferred
            met: Name of the method
            kw_arg: Only the value (not key) of the kwargs dictionary

        Returns:
        """

        if self.obj == obj or obj == getattr(self.obj, "name", None):
            if met == self.method:
                if kw_arg in self.kwargs.values():
                    return True
        return False

    def __str__(self):
        task_str = f"{self.__class__}\n\t" f"Instance: {self.obj.__class__.__name__}\n"
        a = getattr(self.obj, "name", None)
        if a:
            task_str += f"\tName: {a}\n"
        task_str += f"\tMethod: {self.method}\n"
        for i, j in self.kwargs.items():
            task_str += f"\t\t{i}: {j} \n"

        return task_str[:-2]

    def run(self):
        if hasattr(self.obj, "store"):
            self.obj = self.obj.store
        output = getattr(self.obj, self.method)(**self.kwargs)

        if output:
            self.store = output
            del self.obj

        return output

    def __call__(self, *args, **kwargs):
        return self.run()

    def check_exist(self):
        pass


class TaskGraph:
    """
    Context manager of floatcsep workload distribution.

    Assign tasks to a node and defines their dependencies (parent nodes).
    Contains a 'tasks' dictionary whose dict_keys are the Task to be
    executed with dict_values as the Task's dependencies.
    """

    def __init__(self):

        self.tasks = OrderedDict()
        self._ntasks = 0
        self.name = "floatcsep.utils.TaskGraph"

    @property
    def ntasks(self):
        return self._ntasks

    @ntasks.setter
    def ntasks(self, n):
        self._ntasks = n

    def add(self, task):
        """
        Simply adds a defined task to the graph.

        Args:
            task: floatcsep.utils.Task

        Returns:
        """
        self.tasks[task] = []
        self.ntasks += 1

    def add_dependency(self, task, dinst=None, dmeth=None, dkw=None):
        """
        Adds a dependency to a task already inserted to the TaskGraph.

        Searchs
        within the pre-added tasks a signature match by their name/instance,
        method and keyword_args.

        Args:
            task: Task to which a dependency will be asigned
            dinst: object/name of the dependency
            dmeth: method of the dependency
            dkw: keyword argument of the dependency

        Returns:
        """
        deps = []
        for i, other_tasks in enumerate(self.tasks.keys()):
            if other_tasks.sign_match(dinst, dmeth, dkw):
                deps.append(other_tasks)

        self.tasks[task].extend(deps)

    def run(self):
        """
        Iterates through all the graph tasks and runs them.

        Returns:
        """
        for task, deps in self.tasks.items():
            task.run()

    def __call__(self, *args, **kwargs):

        return self.run()

    def check_exist(self):
        pass
