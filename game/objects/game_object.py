from typing import Dict, Optional
from game.components.component import Component
from game.state import State
from game.base import Game
from typing import Dict, Any
from typing import TypeVar, Generic

TC = TypeVar('TC', bound=Component)


class GameObject(Generic[TC]):
    def __init__(self, name: str, 
                       state: State,
                       game: Game):
        self.state = state
        self.game = game
        self.name = name
        self._components: Dict[str, TC] = {}

    def add(self, component: TC) -> 'GameObject':
        #component.on_attach(self)
        self._components[component.name] = component
        return self  # fluent interface

    def get(self, component_name: str) -> Optional[TC]:
        return self._components.get(component_name)

    def update(self, delta_time: float):
        for c in self._components.values():
            if c.enabled:
                c.update(delta_time)

    def render(self):
        pass

T = TypeVar('T', bound=GameObject)

class GameObjectManager(Generic[T]):
    def __init__(self,
                 name: str,
                 state: State,
                 game: Game,
                 collect_gobjects: Dict[Any, T]):
        self.name = name
        self.state = state
        self.game = game
        self.collect_gobjects = collect_gobjects

    def update(self, delta_time: float):
        for k, obj in self.collect_gobjects.items():
            obj.update(delta_time)

    def render(self):
        pass