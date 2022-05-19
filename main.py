def main() -> None:
    knowledge = get_knowledge()

    # live camera feed loop
    # Loop indefinetely
        # if exit key or cross button is pressed, break loop
        # show live camera feed
        # run face detection algorithm
        # if atleast 1 face detected:
            # for each face:
                # run face recognition algorithm against identified_faces database
                # if face is recognized / present in identified_faces database
                    # get known/unknown label from knowledge dictionary
                    # mark face on frame with known/unknown label
                # if face is not recognized / not present in identified_faces database
                    # mark face on frame with unknown label
                    # run face recognition algorithm on each face in unidentified_faces database
                    # if the face is not in unidentified_faces database:
                        # add face to unidentified_faces database
    pass


def get_knowledge() -> dict:
    """
    Returns a dictionary of all the knowledge.json file
    """
    pass


if __name__ == "__main__":
    main()

