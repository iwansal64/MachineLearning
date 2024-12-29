'''This python program is used to decode the dataset into an image'''
import pygame, sys, math, json
from variables import DATASET_FILE_PATH
from typing import List

#? VARIABLE DEFINITIONS
WINDOW_WIDTH = 800                                                                      # Window Width
WINDOW_HEIGHT = 800                                                                     # Window Height

PIXEL_HORIZONTAL = 10                                                                   # How many pixels on the screen at x axis
PIXEL_VERTICAL = 10                                                                     # How many pixels on the screen at y axis
PIXEL_DIMENSIONS = (PIXEL_HORIZONTAL, PIXEL_VERTICAL)                                   # The dimension of the pixel board

PIXEL_WIDTH = 50                                                                        # The width of one pixel
PIXEL_HEIGHT = 50                                                                       # The height of one pixel
PIXEL_SIZE = (PIXEL_WIDTH, PIXEL_HEIGHT)                                                # The size of one pixel (width, height)

PIXEL_GAP = 0                                                                           # The gap between pixels
OFFSET_X = (WINDOW_WIDTH // 2) - ((PIXEL_WIDTH + PIXEL_GAP) * PIXEL_HORIZONTAL // 2)    # The offset of pixel board horizontally (used for centering)
OFFSET_Y = (WINDOW_HEIGHT // 2) - ((PIXEL_HEIGHT + PIXEL_GAP) * PIXEL_VERTICAL // 2)    # The offset of pixel board vertically

PIXEL_CHOOSEN_COLOR = "black"                                                           # A choosen pixel color (activated pixel)
PIXEL_NORMAL_COLOR = "grey"                                                             # A normal pixel color (not activated pixel)
BACKGROUND_COLOR = "white"                                                              # Background color

IMPORT_FROM_DATASET = True                                                              # Import dataset from file?


def main():
    '''Main program'''
    global previous_pixel
            
    # Pygame initializations
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Datasets initializations
    current_datasets = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    if IMPORT_FROM_DATASET:
        with open(DATASET_FILE_PATH, "r+") as f:
            current_datasets = json.loads(f.read())
        
    current_pixels = [[False for j in range(PIXEL_HORIZONTAL)] for i in range(PIXEL_VERTICAL)]
    current_showed_index = 0
    current_showed_number = 1

    def show(dataset: List[bool], normal: bool = True, pixel_horizontal: int = 10):
        

        for i in range(len(dataset)):
            current_pixels[math.floor(i / pixel_horizontal)][i % pixel_horizontal] = dataset[i]
        
        if normal:
            for i in range(len(dataset)):
                for j in range(len(dataset[i])):
                    current_pixels[i][j] = dataset[i][j]
                
        return True

    show([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], False)
    # show(current_datasets[str(current_showed_number)][current_showed_index])
    
    # Clock (used to limit fps)
    clock = pygame.time.Clock()
    
    font = pygame.font.Font(None, 24)
    
    running = True
    while running:
        #? Listen to events
        for event in pygame.event.get():
            #? Quit event
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
                        
            #? Keyboard event
            elif event.type == pygame.KEYDOWN:
                #? Append data
                if event.key == pygame.K_LEFT:
                    current_showed_index -= 1
                    if current_showed_index >= 0:
                        dataset = current_datasets[str(current_showed_number)][current_showed_index]
                        show(dataset)
                    else:
                        current_showed_number -= 1
                        if current_showed_number < 0:
                            current_showed_number = 9
                        current_showed_index = len(current_datasets[str(current_showed_number)]) - 1
                        dataset = current_datasets[str(current_showed_number)][current_showed_index]
                        show(dataset)
                                
                elif event.key == pygame.K_RIGHT:
                    current_showed_index += 1
                    try:
                        dataset = current_datasets[str(current_showed_number)][current_showed_index]
                        show(dataset)
                    except IndexError:
                        current_showed_number += 1
                        if current_showed_number > len(current_datasets.keys()) - 1:
                            current_showed_number = 0
                        current_showed_index = 0
                        dataset = current_datasets[str(current_showed_number)][current_showed_index]
                        show(dataset)

        #? Fill background
        screen.fill(BACKGROUND_COLOR)

        #? Draw the pixels
        for i in range(PIXEL_VERTICAL):
            for j in range(PIXEL_HORIZONTAL):
                pygame.draw.rect(screen, PIXEL_CHOOSEN_COLOR if current_pixels[i][j] else PIXEL_NORMAL_COLOR, (OFFSET_X + (j * (PIXEL_WIDTH + PIXEL_GAP)), OFFSET_Y + (i * (PIXEL_HEIGHT + PIXEL_GAP)), PIXEL_WIDTH, PIXEL_HEIGHT))
        
        text = font.render(f"Data - {current_showed_index} of Number {current_showed_number}", True, (0, 0, 0))
        text_rect = text.get_rect(topleft=(10, 10))
        
        screen.blit(text, text_rect)

        #? Update display
        pygame.display.flip()

        clock.tick(90)
        
        if not running:
            pygame.quit()
        
    
    sys.exit()

    
if __name__ == "__main__":
    main()                                           # The size of one pixel (width, height)