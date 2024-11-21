import pygame
import random

# Initialize pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Catch the Falling Objects")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Basket properties
basket_width = 100
basket_height = 20
basket_x = 350
basket_y = 550

# Object properties
object_width = 20
object_height = 20
object_x = random.randint(0, 780)
object_y = 0
object_speed = 1

# Score
score = 0
font = pygame.font.Font(None, 36)

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the basket with the mouse
    mouse_x, _ = pygame.mouse.get_pos()
    basket_x = mouse_x - basket_width // 2

    # Move the object down
    object_y += object_speed

    # Check if the object is caught by the basket
    if (basket_x < object_x < basket_x + basket_width or basket_x < object_x + object_width < basket_x + basket_width) and basket_y < object_y + object_height < basket_y + basket_height:
        score += 1
        object_x = random.randint(0, 780)
        object_y = 0

    # Check if the object falls off the screen
    if object_y > 600:
        object_x = random.randint(0, 780)
        object_y = 0

    # Clear the screen
    screen.fill(WHITE)

    # Draw the basket
    pygame.draw.rect(screen, BLUE, (basket_x, basket_y, basket_width, basket_height))

    # Draw the falling object
    pygame.draw.rect(screen, BLACK, (object_x, object_y, object_width, object_height))

    # Draw the score
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()