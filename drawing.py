'''This python program is used to draw through pixels and put into dataset'''
import pygame, sys, math, string, json
from variables import DATASET_FILE_PATH
from typing import Dict, List

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
ASK_BEFORE_QUIT = True                                                                  # Ask before quit. If this value is true before quit it'll prompt to terminal wheter you want to save or nah.

previous_pixel = []                                                                     # Previous pixel coordinate (used for preventing double click of a pixel)

def main():
    '''Main program'''
    global previous_pixel
            
    # Pygame initializations
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Datasets initializations
    current_datasets: Dict[str, List[List[int]]] = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    if IMPORT_FROM_DATASET:
        with open(DATASET_FILE_PATH, "r+") as f:
            current_datasets = json.loads(f.read())
        
    # Current pixel (used to keep track of user choosen pixel)
    current_pixels = [[0 for j in range(PIXEL_HORIZONTAL)] for i in range(PIXEL_VERTICAL)]
    # Pressing pixel (used to keep track of the mouse condition. (is it pressing or not))
    pressing_pixels = False
    # Is erasing pixel (used to keep track of wheter user is currently activating or deactivating pixel)
    is_activating_pixel = False

    def save_dataset():
        with open(DATASET_FILE_PATH, "w+") as f:
            f.write(json.dumps(current_datasets))
        print(f"SAVED TO: ({DATASET_FILE_PATH})")


    def touch_pixel(mouse_pos: tuple[int]|list[int], limit: bool = False, mode: bool = False) -> int:
        '''Used to triggering pixel when touched'''
        global previous_pixel
        position = list(mouse_pos)

        #? If it's on the board / pixel area
        if position[0] > OFFSET_X and position[1] > OFFSET_Y and position[0] < WINDOW_WIDTH - OFFSET_X and position[1] < WINDOW_HEIGHT - OFFSET_Y:
            position[0] -= OFFSET_X
            position[1] -= OFFSET_Y

            x_point_refrence = position[0] % (PIXEL_WIDTH + PIXEL_GAP)
            y_point_refrence = position[1] % (PIXEL_HEIGHT + PIXEL_GAP)

            #? If the cursor touched one of the pixel
            if x_point_refrence <= PIXEL_WIDTH and y_point_refrence <= PIXEL_HEIGHT:
                x_index = math.floor(position[0] / (PIXEL_WIDTH + PIXEL_GAP))
                y_index = math.floor(position[1] / (PIXEL_HEIGHT + PIXEL_GAP))

                if previous_pixel != [x_index, y_index] and (current_pixels[y_index][x_index] != mode or not limit):
                    current_pixels[y_index][x_index] = int(not bool(current_pixels[y_index][x_index]))
                    previous_pixel = [x_index, y_index]
                    return current_pixels[y_index][x_index]

        return -1

    # Clock (used to limit fps)
    clock = pygame.time.Clock()

    running = True
    while running:
        #? Listen to events
        for event in pygame.event.get():
            #? Quit event
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            #? Mouse move event
            elif event.type == pygame.MOUSEBUTTONDOWN:
                #? Mouse left click event
                if event.button == 1:
                    res = touch_pixel(event.pos)
                    if res == 0:
                        is_activating_pixel = False
                    elif res == 1:
                        is_activating_pixel = True
                        
                    # If the cursor touch the board
                    if res != -1:
                        pressing_pixels = True
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                #? Mouse left up event
                if event.button == 1:
                    pressing_pixels = False
                    previous_pixel = []
                        
            #? Keyboard event
            elif event.type == pygame.KEYDOWN:
                #? Append data
                if event.unicode in string.digits:
                    current_datasets[event.unicode].append(current_pixels)
                    print(f"Successfully Appending Data to number: {event.unicode}!")
                #? Save dataset
                elif event.key == pygame.K_s:
                    save_dataset()
                #? Reset pixels
                elif event.key == pygame.K_r:
                    current_pixels = [[0 for j in range(PIXEL_HORIZONTAL)] for i in range(PIXEL_VERTICAL)]

        if pressing_pixels:
            touch_pixel(pygame.mouse.get_pos(), True, is_activating_pixel)

        #? Fill background
        screen.fill(BACKGROUND_COLOR)

        #? Draw the pixels
        for i in range(PIXEL_VERTICAL):
            for j in range(PIXEL_HORIZONTAL):
                pygame.draw.rect(screen, PIXEL_CHOOSEN_COLOR if current_pixels[i][j] else PIXEL_NORMAL_COLOR, (OFFSET_X + (j * (PIXEL_WIDTH + PIXEL_GAP)), OFFSET_Y + (i * (PIXEL_HEIGHT + PIXEL_GAP)), PIXEL_WIDTH, PIXEL_HEIGHT))

        #? Update display
        pygame.display.flip()

        clock.tick(90)
        
        if not running:
            pygame.quit()
            if ASK_BEFORE_QUIT and input("Do you want to save the new dataset? [y/n]") == 'y':
                save_dataset()
        
    
    sys.exit()

    
if __name__ == "__main__":
    main()