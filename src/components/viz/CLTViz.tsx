"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";

type Distribution = "uniform" | "exponential" | "bimodal";

function sampleOne(dist: Distribution): number {
  const u = Math.random();
  switch (dist) {
    case "uniform":
      return u * 10;
    case "exponential":
      return -2 * Math.log(1 - u);
    case "bimodal":
      return Math.random() < 0.5 ? d3.randomNormal(3, 0.8)() : d3.randomNormal(7, 0.8)();
    default:
      return u * 10;
  }
}

export default function CLTViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dist, setDist] = useState<Distribution>("uniform");
  const [sampleSize, setSampleSize] = useState(30);
  const [means, setMeans] = useState<number[]>([]);
  const width = 550;
  const height = 350;

  const runSamples = useCallback(
    (count: number) => {
      const newMeans: number[] = [];
      for (let i = 0; i < count; i++) {
        let sum = 0;
        for (let j = 0; j < sampleSize; j++) {
          sum += sampleOne(dist);
        }
        newMeans.push(sum / sampleSize);
      }
      setMeans((prev) => [...prev, ...newMeans]);
    },
    [dist, sampleSize]
  );

  const reset = () => setMeans([]);

  const draw = useCallback(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    if (means.length === 0) {
      g.append("text")
        .attr("x", w / 2)
        .attr("y", h / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "#64748b")
        .attr("font-size", 14)
        .text("Click \"Draw Samples\" to start");
      return;
    }

    // Create histogram
    const xMin = d3.min(means)! - 0.5;
    const xMax = d3.max(means)! + 0.5;
    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, w]);

    const histogram = d3
      .bin()
      .domain([xMin, xMax])
      .thresholds(30)(means);

    const yMax = d3.max(histogram, (d) => d.length)!;
    const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([h, 0]);

    // Draw bars
    histogram.forEach((bin) => {
      if (bin.x0 === undefined || bin.x1 === undefined) return;
      g.append("rect")
        .attr("x", xScale(bin.x0) + 1)
        .attr("y", yScale(bin.length))
        .attr("width", Math.max(0, xScale(bin.x1) - xScale(bin.x0) - 2))
        .attr("height", h - yScale(bin.length))
        .attr("fill", "#818cf8")
        .attr("opacity", 0.7)
        .attr("rx", 1);
    });

    // Overlay normal curve
    if (means.length > 5) {
      const meanVal = d3.mean(means)!;
      const stdVal = d3.deviation(means)!;
      const binWidth = (xMax - xMin) / 30;
      const normalPoints = d3.range(xMin, xMax, (xMax - xMin) / 200).map((x) => ({
        x,
        y:
          (means.length * binWidth) /
          (stdVal * Math.sqrt(2 * Math.PI)) *
          Math.exp(-0.5 * ((x - meanVal) / stdVal) ** 2),
      }));

      const normalLine = d3
        .line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y))
        .curve(d3.curveBasis);

      g.append("path")
        .datum(normalPoints)
        .attr("d", normalLine)
        .attr("fill", "none")
        .attr("stroke", "#f472b6")
        .attr("stroke-width", 2.5);
    }

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(8)).selectAll("text").attr("fill", "#94a3b8");
    g.append("g").call(d3.axisLeft(yScale).ticks(5)).selectAll("text").attr("fill", "#94a3b8");
    g.selectAll("line,path.domain").attr("stroke", "#475569");

    g.append("text").attr("x", w / 2).attr("y", h + 35).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).text("Sample Mean");
    g.append("text").attr("x", -35).attr("y", h / 2).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 12).attr("transform", `rotate(-90,-35,${h / 2})`).text("Count");
  }, [means]);

  useEffect(() => {
    draw();
  }, [draw]);

  const meanVal = means.length > 0 ? d3.mean(means)! : 0;
  const stdVal = means.length > 1 ? d3.deviation(means)! : 0;

  return (
    <div>
      <p className="text-sm text-muted mb-3">
        Sample from a <strong>{dist}</strong> distribution, compute the mean of each sample, and watch the distribution of means become normal — regardless of the original distribution.
      </p>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full max-w-[550px] h-auto rounded-xl border border-border bg-[#0f172a]"
        viewBox={`0 0 ${width} ${height}`}
      />
      <div className="mt-4 flex flex-wrap gap-3">
        <button
          onClick={() => runSamples(100)}
          className="px-4 py-2 text-sm rounded-lg bg-accent text-white font-medium hover:bg-accent-hover transition-colors"
        >
          Draw 100 Samples
        </button>
        <button
          onClick={() => runSamples(1000)}
          className="px-4 py-2 text-sm rounded-lg bg-surface border border-border font-medium hover:bg-surface-hover transition-colors"
        >
          Draw 1000 Samples
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 text-sm rounded-lg bg-surface border border-border font-medium hover:bg-surface-hover transition-colors"
        >
          Reset
        </button>
      </div>
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium block mb-1">Source Distribution:</label>
          <div className="flex flex-wrap gap-2">
            {(["uniform", "exponential", "bimodal"] as const).map((d) => (
              <button
                key={d}
                onClick={() => { setDist(d); setMeans([]); }}
                className={`px-3 py-1 text-xs rounded-lg border transition-colors capitalize ${
                  dist === d ? "bg-accent text-white border-accent" : "bg-surface border-border hover:bg-surface-hover"
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-sm font-medium block mb-1">
            Sample Size (n): {sampleSize}
          </label>
          <input
            type="range"
            min={2}
            max={200}
            value={sampleSize}
            onChange={(e) => { setSampleSize(parseInt(e.target.value)); setMeans([]); }}
            className="w-48"
          />
        </div>
      </div>
      <div className="mt-3 text-xs font-mono text-muted">
        Samples drawn: {means.length} | Mean of means: {meanVal.toFixed(4)} | Std of means: {stdVal.toFixed(4)}
      </div>
    </div>
  );
}
