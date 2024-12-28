import pygame, sys, math, string, json

#? VARIABLE DEFINITIONS
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

PIXEL_HORIZONTAL = 10
PIXEL_VERTICAL = 10
PIXEL_DIMENSIONS = (PIXEL_HORIZONTAL, PIXEL_VERTICAL)

PIXEL_WIDTH = 60
PIXEL_HEIGHT = 60
PIXEL_SIZE = (PIXEL_WIDTH, PIXEL_HEIGHT)

PIXEL_GAP = 5
OFFSET_X = (WINDOW_WIDTH // 2) - ((PIXEL_WIDTH + PIXEL_GAP) * PIXEL_HORIZONTAL // 2)
OFFSET_Y = (WINDOW_HEIGHT // 2) - ((PIXEL_HEIGHT + PIXEL_GAP) * PIXEL_VERTICAL // 2)

PIXEL_CHOOSEN_COLOR = "black"
PIXEL_NORMAL_COLOR = "grey"
BACKGROUND_COLOR = "white"

IMPORT_FROM_DATASET = True
CURRENT_DIRECTORY = "\\".join(__file__.split("\\")[:-1])+"\\"
DATASET_FILE_PATH = CURRENT_DIRECTORY+"dataset.json"

def main():
    '''Main program'''
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    current_datasets = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    print(current_datasets)
    if IMPORT_FROM_DATASET:
        with open(DATASET_FILE_PATH) as f:
            current_datasets = json.loads(f.read())
        

    current_pixels = [[False for j in range(PIXEL_HORIZONTAL)] for i in range(PIXEL_VERTICAL)]
    running = True

    clock = pygame.time.Clock()

    while running:
        #? Listen to events
        for event in pygame.event.get():
            #? Quit event
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                sys.exit()
            #? Mouse move event
            elif event.type == pygame.MOUSEBUTTONDOWN:

                #? Mouse left click event
                if event.button == 1:
                    position = list(event.pos)

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

                            current_pixels[y_index][x_index] = not current_pixels[y_index][x_index]
                            
            #? Keyboard event
            elif event.type == pygame.KEYDOWN:
                if event.unicode in string.digits:
                    current_datasets[event.unicode].append(current_pixels)
                elif event.key == pygame.K_s:
                    with open(DATASET_FILE_PATH, "w+") as f:
                        f.write(json.dumps(current_datasets))
                    print(f"SAVED TO: ({DATASET_FILE_PATH})")

        #? Fill background
        screen.fill(BACKGROUND_COLOR)

        #? Draw the pixels
        for i in range(PIXEL_VERTICAL):
            for j in range(PIXEL_HORIZONTAL):
                pygame.draw.rect(screen, PIXEL_CHOOSEN_COLOR if current_pixels[i][j] else PIXEL_NORMAL_COLOR, (OFFSET_X + (j * (PIXEL_WIDTH + PIXEL_GAP)), OFFSET_Y + (i * (PIXEL_HEIGHT + PIXEL_GAP)), PIXEL_WIDTH, PIXEL_HEIGHT))

        #? Update display
        pygame.display.flip()

        clock.tick(60)

    
if __name__ == "__main__":
    main()