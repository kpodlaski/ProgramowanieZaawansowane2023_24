from game.Game import Game

g = Game()

print(g.state)
print(g.cols)
new_state = g.action(1)
print("state:",new_state)
new_state = g.action(-1)
print("state:",new_state)
new_state = g.action(-1)
print("state:",new_state)
new_state = g.action(-1)
print("state:",new_state)
