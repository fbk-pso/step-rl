from typing import Iterator, Optional, Tuple, List, Dict
import unified_planning as up
import unified_planning.model

class ProblemGenerator:
    def domain(self) -> up.model.Problem:
        raise NotImplementedError

    def size(self) -> Optional[int]:
        raise NotImplementedError

    def get_problems(self) -> Iterator[Tuple[Dict["up.model.FNode", "up.model.FNode"], List["up.model.Object"], List["up.model.FNode"]]]:
        raise NotImplementedError