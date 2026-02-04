# Config keypoint connection rules, events, and instances

Note: this file should be updated before you run Glitter2 to visualize the
tracking results and convert to NIX format.

`cd annolid/configs`

Open `keypoints.yaml` file with a Text Editor.

Here are the default values. Please add the instance names to the `NAME` section.
`EVENTS`, `ZONES`, and other sections can be customized based on your project needs.

```
HEAD:
    #keypoint: keypoint, R,G,B for line color
    nose: "left_ear,102, 204, 255"
    left_ear: "right_ear, 102, 0, 204"
    right_ear: "nose, 51, 102, 255"
BODY:
    tail_base: "left_hip,255, 128, 0"
    Tailbase: "left_hip,255, 128, 0"
    right_hip: "tail_base,153, 255, 204"
    neck: "right_hip,255, 195, 77"
    left_hip: "neck,153, 255, 204"
    frog_m_1: "frog_f_1, 128, 229, 255"

NAME:
    #by default we assume the first item is the subject animal name
    # the second row for left interact object name
    # the third row for the right interact object name
    Mouse
    LeftTeaball
    RightTeaball
    resident
    mouse
    intruder
    vole
    P6_Huddle
    P6_Lone
    frog_m_1
    frog_f_1
    frog_m_2
    frog_f_2

EVENTS:
    huddling
    grooming
    rearing
    sniffing
    investigation
    LeftInteract
    RightInteract
    nose_to_nose
    fighting
    flank_sniffing
    chasing
    anogenital_sniffing
    walking
    sitting
    attack
    pounce
    chase
    running
    running_away
    exploring
    immobility
    autogrooming
    nose_to_nose_sniffing
    intruder_rearing

ZONES:
    LeftZone
    RightZone
```

Save the updated file.
