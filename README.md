# autonomous-turtle
Thesis Project by Gargi Sadalgekar, Jacob Walrath, and Samarie Wilson.

## Problem Definition
A central property of multirobot systems is how the system assigns work and coordinates cooperative effort
between individual robots. This project will aim to implement and demonstrate an approach to the task
allocation problem in a multirobot system in the presence of uncertainty. The multirobot system will be
comprised of "turtlebot" units equipped with different classes of manipulator arms representing the ability
to retrieve specific types of objects. The turtlebot system's objective will be to clear a field of objects, where
each object will require either a specific manipulator or the joint effort of multiple robots to be retrieved.

Thus, successful completion of this task would represent the ability to perform effective decision making
as well as efficiently allocate both specialized and coordinated resources in an uncertain environment.

## Methodology
The execution of this project will be split into two phases corresponding to the fall and spring semesters,
with the hope that in-person access to departmental labs, resources, and other facilities will be an option
in the spring but with a contingency option in the case that the entirety of the project must be conducted
remotely.

Phase one will start with initial manipulator and object design and culminate in the development of
the system's task allocation decision making algorithms in simulation. Phase two, if in-person, will be the
implementation and testing of the physical turtlebots. If virtual, hardware implementation will be facilitated
through staff/faculty present on campus with access to the necessary facilities, or left in favor of doing further
refinement in the simulation environment.

Ultimately, the multirobot system and field clearance task will be constructed such that:
1. The multirobot system is **heterogeneous**: The turtlebots will be outfitted with one of several manip-
ulator arms, so individual units will be purpose-built to retrieve one class of objects while the whole
"bale" of turtlebots together will be able to address objects from any of several types.
2. The multirobot system is asked to solve a **task allocation problem**: The total field will be pre-
divided into discrete areas, with tasks defined as sensing and collecting objects from a single area. The
bale of turtlebots will then be tasked with clearing the several areas simultaneously.
3. The multirobot system is asked to **make decisions under uncertainty**: Because the location and
number of objects for retrieval is unknown prior to start, the turtlebot bale will utilize multi-armed
bandit algorithms to efficiently find objects and identify what kind/how many turtlebots to assign to
clear them.
4. The multirobot system will be required to **coordinate cooperative effort**: Certain objects will be
constructed such that the use of multiple turtlebots is necessary to transport them once identified.
