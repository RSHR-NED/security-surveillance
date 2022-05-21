## identified_faces/

Database for images of faces that have been identified, i.e. marked as known/unkown by user. Each image file is to be named with the following format: `<id of face>.jpg`(or .png, extension doesn't matter).

## unidentified_faces/

Database for images of faces that have not been identified yet. `identify.py` contains logic for marking face images in this folder as known/unkown.

## knowledge.json

Contains mapping of face IDs to "known" or "unkown" label.
eg:

```
{
    "1": "known",
    "2": "known",
    "3": "unknown"
}
```

## main.py

Contains logic for camera feed, face detection, face recogniton.

##### get_knowledge()

Reads knowledge.json and returns in python dictionary form.

##### main()

Rough pseudocode:

```
live camera feed loop
    Loop indefinetely
        if exit key or cross button is pressed, break loop
        show live camera feed
        run face detection algorithm
        if atleast 1 face detected:
            for each face:
                run face recognition algorithm against identified_faces database
                if face is recognized / present in identified_faces database
                    get known/unknown label from knowledge dictionary
                    mark face on frame with known/unknown label
                if face is not recognized / not present in identified_faces database
                    mark face on frame with unknown label
                    run face recognition algorithm on each face in unidentified_faces database
                    if the face is not in unidentified_faces database:
                        add face to unidentified_faces database
```

## identify.py

A GUI application for user to mark all face images in `unidentified_faces` database as known or unknown. User should be able to view all the faces in the database one by one and mark them as known or unknown using buttons. knowledge.json` will be updated accordingly, and marked face would be moved to identified_faces database. If no face is presesent in the database then appropriate message is to be displayed.
