"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

export default function AttentionViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedRow, setSelectedRow] = useState(0);
  const [temperature, setTemperature] = useState(1.0);
  const width = 600;
  const height = 500;

  const tokens = ["The", "cat", "sat", "on", "the", "mat", "[EOS]"];
  const n = tokens.length;

  // Simulate Q, K matrices for attention — these create intuitive patterns
  // where content words attend to related content words
  const baseScores = [
    //  The   cat   sat    on   the   mat   EOS
    [0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1], // The -> spread
    [0.1, 0.3, 0.15, 0.05, 0.1, 0.25, 0.05], // cat -> cat, sat, mat
    [0.05, 0.3, 0.2, 0.15, 0.05, 0.2, 0.05], // sat -> cat, sat, on, mat
    [0.1, 0.05, 0.2, 0.15, 0.1, 0.3, 0.1], // on -> sat, mat
    [0.25, 0.1, 0.1, 0.1, 0.2, 0.15, 0.1], // the -> The, the, mat
    [0.05, 0.25, 0.2, 0.2, 0.1, 0.15, 0.05], // mat -> cat, sat, on
    [0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.25], // EOS -> spread + self
  ];

  // Apply temperature-scaled softmax
  const softmax = useCallback(
    (scores: number[]) => {
      const scaled = scores.map((s) => s / temperature);
      const maxS = Math.max(...scaled);
      const exps = scaled.map((s) => Math.exp(s - maxS));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / sum);
    },
    [temperature]
  );

  const attentionWeights = baseScores.map((row) => softmax(row));

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 60, right: 20, bottom: 20, left: 80 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const cellSize = Math.min(w / n, h / n);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3
      .scaleSequential(d3.interpolateYlOrRd)
      .domain([0, d3.max(attentionWeights.flat())!]);

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = attentionWeights[i][j];
        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 1)
          .attr("height", cellSize - 1)
          .attr("rx", 3)
          .attr("fill", colorScale(val))
          .attr("opacity", selectedRow === -1 || selectedRow === i ? 1 : 0.2)
          .attr("stroke", selectedRow === i && j === d3.maxIndex(attentionWeights[i]) ? "#fff" : "none")
          .attr("stroke-width", 2)
          .attr("cursor", "pointer")
          .on("mouseover", function () {
            d3.select(this).attr("stroke", "#fff").attr("stroke-width", 1);
            tooltip
              .style("opacity", 1)
              .html(
                `<strong>${tokens[i]}</strong> → <strong>${tokens[j]}</strong><br/>Weight: ${val.toFixed(4)}`
              )
              .style("left", `${j * cellSize + margin.left + cellSize / 2}px`)
              .style("top", `${i * cellSize + margin.top - 10}px`);
          })
          .on("mouseout", function () {
            d3.select(this)
              .attr("stroke", selectedRow === i && j === d3.maxIndex(attentionWeights[i]) ? "#fff" : "none")
              .attr("stroke-width", selectedRow === i && j === d3.maxIndex(attentionWeights[i]) ? 2 : 0);
            tooltip.style("opacity", 0);
          });

        // Value text
        if (val > 0.05) {
          g.append("text")
            .attr("x", j * cellSize + cellSize / 2)
            .attr("y", i * cellSize + cellSize / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", val > 0.18 ? "#fff" : "#1e293b")
            .attr("font-size", 10)
            .attr("font-weight", "bold")
            .attr("pointer-events", "none")
            .text(val.toFixed(2));
        }
      }
    }

    // Column headers (Keys)
    tokens.forEach((token, j) => {
      g.append("text")
        .attr("x", j * cellSize + cellSize / 2)
        .attr("y", -8)
        .attr("text-anchor", "middle")
        .attr("fill", "#94a3b8")
        .attr("font-size", 13)
        .attr("font-weight", "bold")
        .text(token);
    });

    // Column label
    g.append("text")
      .attr("x", (n * cellSize) / 2)
      .attr("y", -35)
      .attr("text-anchor", "middle")
      .attr("fill", "#64748b")
      .attr("font-size", 11)
      .text("Keys (attending to)");

    // Row headers (Queries)
    tokens.forEach((token, i) => {
      g.append("text")
        .attr("x", -10)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "central")
        .attr("fill", selectedRow === i ? "#818cf8" : "#94a3b8")
        .attr("font-size", 13)
        .attr("font-weight", selectedRow === i ? "bold" : "normal")
        .attr("cursor", "pointer")
        .text(token)
        .on("click", () => setSelectedRow(i));
    });

    // Row label
    g.append("text")
      .attr("x", -45)
      .attr("y", (n * cellSize) / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#64748b")
      .attr("font-size", 11)
      .attr("transform", `rotate(-90,-45,${(n * cellSize) / 2})`);

    // Tooltip
    const tooltip = d3
      .select(svgRef.current!.parentElement!)
      .selectAll(".attn-tooltip")
      .data([0])
      .join("div")
      .attr("class", "attn-tooltip")
      .style("position", "absolute")
      .style("pointer-events", "none")
      .style("background", "#1e293b")
      .style("color", "#e2e8f0")
      .style("padding", "6px 10px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("opacity", 0)
      .style("transition", "opacity 0.15s")
      .style("z-index", "10");
  }, [attentionWeights, selectedRow, temperature]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div>
      <div className="relative">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="w-full max-w-[600px] h-auto rounded-xl border border-border bg-[#0f172a]"
          viewBox={`0 0 ${width} ${height}`}
        />
      </div>
      <div className="mt-4 space-y-3">
        <div>
          <label className="text-sm font-medium block mb-1">
            Focus on query token:
          </label>
          <div className="flex flex-wrap gap-2">
            {tokens.map((t, i) => (
              <button
                key={i}
                onClick={() => setSelectedRow(i)}
                className={`px-3 py-1 text-xs rounded-lg border transition-colors ${
                  selectedRow === i
                    ? "bg-accent text-white border-accent"
                    : "bg-surface border-border hover:bg-surface-hover"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-sm font-medium block mb-1">
            Temperature: {temperature.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-48"
          />
          <p className="text-xs text-muted mt-1">
            Lower temperature → sharper attention (focuses on top keys).
            Higher → more uniform attention.
          </p>
        </div>
      </div>
    </div>
  );
}
