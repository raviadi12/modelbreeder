import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import * as tf from "@tensorflow/tfjs";

const ModelVisualizer = ({ model, animationPlay }) => {
  const mountRef = useRef(null);
  const overlayRef = useRef(null); // Ref for the text overlay

  const getModelDescription = (model) => {
    const description = [];
    description.push(`Model: ${model.name || 'Unnamed Model'}`);
    description.push(`Tensor Input Size: ${model.inputs[0].shape.slice(1).join('x')}`);
    
    model.layers.forEach((layer, index) => {
      let layerDescription = `Layer ${index + 1}: ${layer.name} (`;
  
      if (layer instanceof tf.layers.conv2d) {
        layerDescription += `Conv2D, ${layer.filters} filters, ${layer.kernelSize[0]}x${layer.kernelSize[1]} kernel, ${layer.activation} activation`;
      } else if (layer instanceof tf.layers.maxPooling2d) {
        layerDescription += `MaxPooling2D, ${layer.poolSize[0]}x${layer.poolSize[1]} pool size`;
      } else if (layer instanceof tf.layers.dropout) {
        layerDescription += `Dropout, rate ${layer.rate}`;
      } else if (layer instanceof tf.layers.dense) {
        layerDescription += `Dense, ${layer.units} units, ${layer.activation} activation`;
      } else if (layer instanceof tf.layers.flatten) {
        layerDescription += `Flatten`;
      } else if (layer instanceof tf.layers.batchNormalization) {
        layerDescription += `BatchNormalization`;
      } else {
        layerDescription += `Unknown Layer Type`;
      }
  
      layerDescription += ')';
      description.push(layerDescription);
    });
  
    return description.join('<br />');
  };

  // State variables for mouse movement
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const [startY, setStartY] = useState(0);
  const [initialAngle, setInitialAngle] = useState(0);
  const [renderLine, setRenderLine] = useState(false);
  const [initialYAngle, setInitialYAngle] = useState(0);

  useEffect(() => {
    const mount = mountRef.current;
    const overlay = overlayRef.current;
    

    // Create scene, camera, and renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      55,
      mount.clientWidth / mount.clientHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); // Set alpha to true
    renderer.setClearColor("0x0000FF", 1); // Set clearAlpha to 0 (the second argument)
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true; // Enable shadow maps
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Optional: choose shadow map type
    mount.appendChild(renderer.domElement);

    let angle = 0;
    let radius = 100;
    const center = new THREE.Vector3(60, 0, 0);
    let yangle = -50;

    // Create directional light to simulate sunlight
    const directionalLight = new THREE.DirectionalLight(0xffffff, 3);
    directionalLight.position.set(50, 50, 50);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048; // Increase shadow map resolution if needed
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.camera.left = -50;
    directionalLight.shadow.camera.right = 50;
    directionalLight.shadow.camera.top = 50;
    directionalLight.shadow.camera.bottom = -50;
    scene.add(directionalLight);

    // Add ambient light for some basic lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 8); // soft white light
    scene.add(ambientLight);

    // Function to create a cube
    const createCube = (width, height, depth, color) => {
      const geometry = new THREE.BoxGeometry(width, height, depth);
      const material = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.2, // Adjust this value between 0 and 1
        roughness: 0.5, // Adjust this value between 0 and 1
      });
      const cube = new THREE.Mesh(geometry, material);
      cube.castShadow = true; // Enable shadows for the cube
      cube.receiveShadow = true; // Enable the cube to receive shadows
      return cube;
    };

    // Function to visualize the model layers
    const visualizeModel = (model) => {
      const layers = model.layers;
      let xOffset = -layers.length * 1.5; // Initial offset

      // Helper function to create a cube
      const createCube = (width, height, depth, color) => {
        const geometry = new THREE.BoxGeometry(width, height, depth);
        const material = new THREE.MeshBasicMaterial({ color });
        
        const cube = new THREE.Mesh(geometry, material);
        cube.castShadow = true; // Enable shadows for the cube
        cube.receiveShadow = true; // Enable the cube to receive shadows
        return cube;
      };

      // Helper function to create a line between two points
      const createLine = (start, end, color) => {
        const material = new THREE.LineBasicMaterial({ color });
        const points = [start, end];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        return new THREE.Line(geometry, material);
      };

      const previousLayerCubes = [];

      layers.forEach((layer, index) => {
        const layerSize =
          layer.outputShape.slice(1).reduce((a, b) => a * b, 1) ** 0.5;
        const color = new THREE.Color(
          `hsl(${(index / layers.length) * 360}, 100%, 50%)`
        );
        const cubeSize = layerSize / 20;
        const cube = createCube(cubeSize, cubeSize, cubeSize, color);
        cube.position.x = xOffset;
        xOffset += cubeSize * 1.5; // Space based on the layer size
        scene.add(cube);

        // Add lines connecting outer vertices to the previous layer cubes
        if (previousLayerCubes.length > 0) {
          previousLayerCubes.forEach((prevCube) => {
            // Define the corners of the previous and current cubes
            const prevCorners = [
              new THREE.Vector3(
                prevCube.position.x - prevCube.geometry.parameters.width / 2,
                prevCube.position.y - prevCube.geometry.parameters.height / 2,
                prevCube.position.z - prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x - prevCube.geometry.parameters.width / 2,
                prevCube.position.y - prevCube.geometry.parameters.height / 2,
                prevCube.position.z + prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x - prevCube.geometry.parameters.width / 2,
                prevCube.position.y + prevCube.geometry.parameters.height / 2,
                prevCube.position.z - prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x - prevCube.geometry.parameters.width / 2,
                prevCube.position.y + prevCube.geometry.parameters.height / 2,
                prevCube.position.z + prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x + prevCube.geometry.parameters.width / 2,
                prevCube.position.y - prevCube.geometry.parameters.height / 2,
                prevCube.position.z - prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x + prevCube.geometry.parameters.width / 2,
                prevCube.position.y - prevCube.geometry.parameters.height / 2,
                prevCube.position.z + prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x + prevCube.geometry.parameters.width / 2,
                prevCube.position.y + prevCube.geometry.parameters.height / 2,
                prevCube.position.z - prevCube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                prevCube.position.x + prevCube.geometry.parameters.width / 2,
                prevCube.position.y + prevCube.geometry.parameters.height / 2,
                prevCube.position.z + prevCube.geometry.parameters.depth / 2
              ),
            ];

            const currentCorners = [
              new THREE.Vector3(
                cube.position.x - cube.geometry.parameters.width / 2,
                cube.position.y - cube.geometry.parameters.height / 2,
                cube.position.z - cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x - cube.geometry.parameters.width / 2,
                cube.position.y - cube.geometry.parameters.height / 2,
                cube.position.z + cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x - cube.geometry.parameters.width / 2,
                cube.position.y + cube.geometry.parameters.height / 2,
                cube.position.z - cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x - cube.geometry.parameters.width / 2,
                cube.position.y + cube.geometry.parameters.height / 2,
                cube.position.z + cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x + cube.geometry.parameters.width / 2,
                cube.position.y - cube.geometry.parameters.height / 2,
                cube.position.z - cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x + cube.geometry.parameters.width / 2,
                cube.position.y - cube.geometry.parameters.height / 2,
                cube.position.z + cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x + cube.geometry.parameters.width / 2,
                cube.position.y + cube.geometry.parameters.height / 2,
                cube.position.z - cube.geometry.parameters.depth / 2
              ),
              new THREE.Vector3(
                cube.position.x + cube.geometry.parameters.width / 2,
                cube.position.y + cube.geometry.parameters.height / 2,
                cube.position.z + cube.geometry.parameters.depth / 2
              ),
            ];
            if (renderLine) {
            // Create lines between corresponding corners
            for (let i = 0; i < prevCorners.length; i++) {
              const line = createLine(prevCorners[i], currentCorners[i], color);
              scene.add(line);
            }
          }
          });
        }

        previousLayerCubes.push(cube); // Store the current cube for the next layer
      });
    };

    // Visualize the model
    if (model) {
      visualizeModel(model);
    }

    // Animation loop
    const animate = () => {
      if (animationPlay) {
        requestAnimationFrame(animate);
      }
      if (!isDragging) {
        angle += 0.01; // Increment the angle
      //  yangle += 0.01;
      }
      camera.position.x = radius * Math.cos(angle) + 60; // Calculate x position in polar coordinates
      camera.position.z = radius * Math.sin(angle); // Calculate z position in polar coordinates
      camera.position.y = radius * Math.sin(yangle);
      camera.lookAt(center); // Ensure the camera always points to the center
      renderer.render(scene, camera);
    };

    if (animationPlay) {
      animate();
    }

    const handleResize = () => {
      overlay.style.left = `${mount.offsetLeft}px`;
      overlay.style.top = `${mount.offsetTop}px`;
    };
    window.addEventListener("resize", handleResize);
    handleResize(); // Set initial position

    // Mouse events
    const handleMouseDown = (event) => {
      setIsDragging(true);
      setStartX(event.clientX);
      setStartY(event.clientY);
      setInitialAngle(angle);
    };
    
    const handleMouseMove = (event) => {
      if (isDragging) {
        const deltaX = event.clientX - startX;
        const deltaY = event.clientY - startY;
        angle = initialAngle + deltaX * 0.01; // Adjust the sensitivity as needed
        yangle = initialAngle + deltaY * 0.01; // Adjust the sensitivity as needed
      }
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
    };

    let minRadius = 30;
    let maxRadius = 300;

    if (overlayRef.current) {
      overlayRef.current.innerHTML = getModelDescription(model);
    }

    const handleWheel = (event) => {
      event.preventDefault(); // Prevent default scrolling behavior
      radius += event.deltaY * 0.1; // Adjust the sensitivity as needed
      // Ensure radius stays within a certain range
      radius = Math.max(radius, minRadius); // Minimum radius
      radius = Math.min(radius, maxRadius); // Maximum radius
    };
    
    mount.addEventListener("mousedown", handleMouseDown);
    mount.addEventListener("mousemove", handleMouseMove);
    mount.addEventListener("mouseup", handleMouseUp);
    mount.addEventListener("wheel", handleWheel);

    // Cleanup
    return () => {
      mount.removeChild(renderer.domElement);
      mount.removeEventListener("mousedown", handleMouseDown);
      mount.removeEventListener("mousemove", handleMouseMove);
      mount.removeEventListener("mouseup", handleMouseUp);
      mount.removeEventListener("wheel", handleWheel);
      window.removeEventListener("resize", handleResize);
    };
  }, [model, animationPlay, isDragging, startX, startY, initialAngle, renderLine]);

  return (
    <div>
      <div
        ref={overlayRef}
        style={{
          position: "absolute",
          fontFamily: "Arial, sans-serif",
          fontSize: "9px",
          color: "white",
          backgroundColor: "rgba(0, 0, 0, 0.5)",
          padding: "4px 8px",
          borderRadius: "4px",
          textAlign: "left",
        }}
      ></div>
        <button onClick={() => renderLine ? setRenderLine(false) : setRenderLine(true)} style={{width:'150px', fontSize:'14px', height:'26px'}}>Render Line</button>
      <div
        ref={mountRef}
        style={{ width: "50vw", height: "380px", backgroundColor: "black" }}
      />
    </div>
  );
};

export default ModelVisualizer;
