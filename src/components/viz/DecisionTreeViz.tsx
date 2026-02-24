"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

interface DataPoint {
  x: number;
  y: number;
  label: number;
}

export default function DecisionTreeViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [maxDepth, setMaxDepth] = useState(3);
  const width = 550;
  const height = 400;

  // Generate some 2D classification data
  const data: DataPoint[] = [
    // Cluster 0 (bottom-left)
    { x: 1.2, y: 1.5, label: 0 }, { x: 1.8, y: 2.1, label: 0 },
    { x: 2.3, y: 1.0, label: 0 }, { x: 1.5, y: 2.5, label: 0 },
    { x: 2.0, y: 1.8, label: 0 }, { x: 2.5, y: 2.8, label: 0 },
    { x: 1.0, y: 1.0, label: 0 }, { x: 2.8, y: 1.5, label: 0 },
    { x: 1.3, y: 3.0, label: 0 }, { x: 2.2, y: 2.3, label: 0 },
    // Cluster 1 (top-right)
    { x: 5.5, y: 6.0, label: 1 }, { x: 6.2, y: 7.1, label: 1 },
    { x: 5.0, y: 5.5, label: 1 }, { x: 6.8, y: 6.5, label: 1 },
    { x: 5.8, y: 7.5, label: 1 }, { x: 7.0, y: 7.0, label: 1 },
    { x: 5.3, y: 6.8, label: 1 }, { x: 6.5, y: 5.8, label: 1 },
    { x: 7.2, y: 6.2, label: 1 }, { x: 6.0, y: 8.0, label: 1 },
    // Some mixed region
    { x: 3.5, y: 4.0, label: 0 }, { x: 4.0, y: 4.5, label: 1 },
    { x: 3.8, y: 3.5, label: 0 }, { x: 4.2, y: 5.0, label: 1 },
    { x: 3.0, y: 4.2, label: 0 }, { x: 4.5, y: 3.8, label: 1 },
  ];

  // Simple decision tree implementation
  interface TreeNode {
    feature?: "x" | "y";
    threshold?: number;
    left?: TreeNode;
    right?: TreeNode;
    prediction?: number;
    depth: number;
    gini?: number;
  }

  const gini = (labels: number[]): number => {
    if (labels.length === 0) return 0;
    const counts = [0, 0];
    labels.forEach((l) => counts[l]++);
    const p0 = counts[0] / labels.length;
    const p1 = counts[1] / labels.length;
    return 1 - p0 * p0 - p1 * p1;
  };

  const buildTree = useCallback(
    (pts: DataPoint[], depth: number): TreeNode => {
      if (depth >= maxDepth || pts.length <= 2) {
        const counts = [0, 0];
        pts.forEach((p) => counts[p.label]++);
        return { prediction: counts[1] >= counts[0] ? 1 : 0, depth, gini: gini(pts.map(p => p.label)) };
      }

      let bestGini = Infinity;
      let bestFeature: "x" | "y" = "x";
      let bestThreshold = 0;

      for (const feature of ["x", "y"] as const) {
        const values = [...new Set(pts.map((p) => p[feature]))].sort((a, b) => a - b);
        for (let i = 0; i < values.length - 1; i++) {
          const threshold = (values[i] + values[i + 1]) / 2;
          const left = pts.filter((p) => p[feature] <= threshold);
          const right = pts.filter((p) => p[feature] > threshold);
          const weightedGini =
            (left.length / pts.length) * gini(left.map((p) => p.label)) +
            (right.length / pts.length) * gini(right.map((p) => p.label));
          if (weightedGini < bestGini) {
            bestGini = weightedGini;
            bestFeature = feature;
            bestThreshold = threshold;
          }
        }
      }

      const leftPts = pts.filter((p) => p[bestFeature] <= bestThreshold);
      const rightPts = pts.filter((p) => p[bestFeature] > bestThreshold);

      if (leftPts.length === 0 || rightPts.length === 0) {
        const counts = [0, 0];
        pts.forEach((p) => counts[p.label]++);
        return { prediction: counts[1] >= counts[0] ? 1 : 0, depth, gini: bestGini };
      }

      return {
        feature: bestFeature,
        threshold: bestThreshold,
        left: buildTree(leftPts, depth + 1),
        right: buildTree(rightPts, depth + 1),
        depth,
        gini: bestGini,
      };
    },
    [maxDepth]
  );

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 8.5]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 9]).range([h, 0]);

    const tree = buildTree(data, 0);

    // Draw decision boundaries by coloring the background
    const resolution = 2;
    const predict = (node: TreeNode, x: number, y: number): number => {
      if (node.prediction !== undefined) return node.prediction;
      if (!node.feature || node.threshold === undefined) return 0;
      const val = node.feature === "x" ? x : y;
      if (val <= node.threshold) return predict(node.left!, x, y);
      return predict(node.right!, x, y);
    };

    for (let px = 0; px < w; px += resolution) {
      for (let py = 0; py < h; py += resolution) {
        const x = xScale.invert(px);
        const y = yScale.invert(py);
        const pred = predict(tree, x, y);
        g.append("rect")
          .attr("x", px)
          .attr("y", py)
          .attr("width", resolution)
          .attr("height", resolution)
          .attr("fill", pred === 0 ? "#818cf8" : "#f472b6")
          .attr("opacity", 0.15);
      }
    }

    // Draw split lines
    const drawSplits = (node: TreeNode, xMin: number, xMax: number, yMin: number, yMax: number) => {
      if (node.prediction !== undefined) return;
      if (!node.feature || node.threshold === undefined) return;

      if (node.feature === "x") {
        g.append("line")
          .attr("x1", xScale(node.threshold))
          .attr("y1", yScale(yMin))
          .attr("x2", xScale(node.threshold))
          .attr("y2", yScale(yMax))
          .attr("stroke", "#f59e0b")
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", "6,3");
        if (node.left) drawSplits(node.left, xMin, node.threshold, yMin, yMax);
        if (node.right) drawSplits(node.right, node.threshold, xMax, yMin, yMax);
      } else {
        g.append("line")
          .attr("x1", xScale(xMin))
          .attr("y1", yScale(node.threshold))
          .attr("x2", xScale(xMax))
          .attr("y2", yScale(node.threshold))
          .attr("stroke", "#f59e0b")
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", "6,3");
        if (node.left) drawSplits(node.left, xMin, xMax, yMin, node.threshold);
        if (node.right) drawSplits(node.right, xMin, xMax, node.threshold, yMax);
      }
    };
    drawSplits(tree, 0, 8.5, 0, 9);

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xScale)).selectAll("text").attr("fill", "#94a3b8");
    g.append("g").call(d3.axisLeft(yScale)).selectAll("text").attr("fill", "#94a3b8");
    g.selectAll("line,path.domain").attr("stroke", "#475569");

    // Data points
    data.forEach((p) => {
      g.append("circle")
        .attr("cx", xScale(p.x))
        .attr("cy", yScale(p.y))
        .attr("r", 5)
        .attr("fill", p.label === 0 ? "#818cf8" : "#f472b6")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5);
    });

    // Labels
    g.append("text").attr("x", w / 2).attr("y", h + 35).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).text("Feature x₁");
    g.append("text").attr("x", -35).attr("y", h / 2).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).attr("transform", `rotate(-90,-35,${h / 2})`).text("Feature x₂");
  }, [maxDepth, buildTree]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[550px] h-auto rounded-xl border border-border bg-[#0f172a]"
        viewBox={`0 0 ${width} ${height}`}
      />
      <div className="mt-4">
        <label className="text-sm font-medium block mb-1">
          Max Depth: {maxDepth}
        </label>
        <input
          type="range"
          min={1}
          max={8}
          value={maxDepth}
          onChange={(e) => setMaxDepth(parseInt(e.target.value))}
          className="w-48"
        />
        <p className="text-xs text-muted mt-2">
          Increase depth to see more splits. Notice how deeper trees create more complex (potentially overfit) boundaries.
        </p>
      </div>
      <div className="flex gap-4 mt-3 text-xs">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-[#818cf8]" /> Class 0
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-[#f472b6]" /> Class 1
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-1 bg-[#f59e0b]" /> Split boundary
        </span>
      </div>
    </div>
  );
}
