import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class BenchmarkPlot:
    def __init__(self, filepath: str = "llm_inference_benchmark.csv", output_dir: str = "plots"):
        self.filepath = filepath
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_data(self):
        return pd.read_csv(self.filepath)

    def plot_ttft(self):
        """
        Plot average Time to First Token (TTFT) vs total requests for different schedulers.

        Groups data by scheduler_type and run_num, then plots:
        - X-axis: max(seq) - total number of requests in each run
        - Y-axis: average TTFT - average time to first token
        """
        df = self.get_data()

        # Group by scheduler_type and run_num, then aggregate
        agg_df = (df
                  .groupby(["scheduler_type", "run_num"])
                  .agg({
                      "seq": "max",           # Total requests size (max seq number)
                      "ttft": "mean"          # Average time to first token
                  })
                  .reset_index()
                )

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting ...\n\n")

        # Pivot data for grouped bar chart
        pivot_df = agg_df.pivot(index="seq", columns="scheduler_type", values="ttft")

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up bar positions
        x = np.arange(len(pivot_df.index))
        width = 0.25  # Width of each bar
        schedulers = pivot_df.columns.tolist()

        # Plot bars for each scheduler
        for i, scheduler in enumerate(schedulers):
            offset = width * (i - len(schedulers)/2 + 0.5)
            bars = ax.bar(
                x + offset,
                pivot_df[scheduler],
                width,
                label=scheduler,
                alpha=0.8
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        rotation=0
                    )

        # Customize the plot
        ax.set_xlabel("Total Requests (max seq)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Time to First Token (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Average TTFT by Scheduler Type and Request Count", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend(title="Scheduler Type", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "ttft_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return agg_df

    def plot_throughput(self):
        """
        Plot average throughput (tokens per second) for different schedulers.

        Groups data by scheduler_type and run_num, then plots:
        - X-axis: max(seq) - total number of requests in each run
        - Y-axis: average tokens per second - throughput metric
        """
        df = self.get_data()

        # Group by scheduler_type and run_num, then aggregate
        agg_df = (df
                  .groupby(["scheduler_type", "run_num"])
                  .agg({
                      "seq": "max",              # Total requests size
                      "tokens_per_sec": "mean"   # Average throughput
                  })
                  .reset_index()
                )

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting Throughput...\n\n")

        # Pivot data for grouped bar chart
        pivot_df = agg_df.pivot(index="seq", columns="scheduler_type", values="tokens_per_sec")

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up bar positions
        x = np.arange(len(pivot_df.index))
        width = 0.25  # Width of each bar
        schedulers = pivot_df.columns.tolist()

        # Plot bars for each scheduler
        for i, scheduler in enumerate(schedulers):
            offset = width * (i - len(schedulers)/2 + 0.5)
            bars = ax.bar(
                x + offset,
                pivot_df[scheduler],
                width,
                label=scheduler,
                alpha=0.8
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        rotation=0
                    )

        # Customize the plot
        ax.set_xlabel("Total Requests (max seq)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Throughput (tokens/sec)", fontsize=12, fontweight='bold')
        ax.set_title("Average Throughput by Scheduler Type and Request Count", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend(title="Scheduler Type", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "throughput_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return agg_df

    def plot_total_latency(self):
        """
        Plot total latency (sum of all latencies) for different schedulers.

        This shows the total time to complete ALL requests in each run.
        - X-axis: max(seq) - total number of requests in each run
        - Y-axis: sum of latencies - total completion time
        """
        df = self.get_data()

        # Group by scheduler_type and run_num, then aggregate
        agg_df = (df
                  .groupby(["scheduler_type", "run_num"])
                  .agg({
                      "seq": "max",        # Total requests size
                      "latency": "sum"     # Total completion time
                  })
                  .reset_index()
                )

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting Total Latency...\n\n")

        # Pivot data for grouped bar chart
        pivot_df = agg_df.pivot(index="seq", columns="scheduler_type", values="latency")

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set up bar positions
        x = np.arange(len(pivot_df.index))
        width = 0.25  # Width of each bar
        schedulers = pivot_df.columns.tolist()

        # Plot bars for each scheduler
        for i, scheduler in enumerate(schedulers):
            offset = width * (i - len(schedulers)/2 + 0.5)
            bars = ax.bar(
                x + offset,
                pivot_df[scheduler],
                width,
                label=scheduler,
                alpha=0.8
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.1f}s',
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        rotation=0
                    )

        # Customize the plot
        ax.set_xlabel("Total Requests (max seq)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Total Completion Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Total Latency by Scheduler Type and Request Count", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend(title="Scheduler Type", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "total_latency_bar_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return agg_df

    def plot_ttft_line(self):
        """
        Plot average TTFT as line chart for different schedulers.
        Line charts are better for showing trends across request counts.
        """
        df = self.get_data()

        # Group by scheduler_type and run_num, then aggregate
        agg_df = (df
                  .groupby(["scheduler_type", "run_num"])
                  .agg({
                      "seq": "max",
                      "ttft": "mean"
                  })
                  .reset_index()
                )

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting TTFT Line Chart...\n\n")

        # Create the line chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each scheduler type with different line style
        for scheduler in agg_df["scheduler_type"].unique():
            scheduler_data = agg_df[agg_df["scheduler_type"] == scheduler].sort_values("seq")
            ax.plot(
                scheduler_data["seq"],
                scheduler_data["ttft"],
                marker='o',
                label=scheduler,
                linewidth=2.5,
                markersize=10
            )

        # Customize the plot
        ax.set_xlabel("Total Requests (max seq)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Time to First Token (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("TTFT Trend Across Request Loads", fontsize=14, fontweight='bold')
        ax.legend(title="Scheduler Type", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "ttft_line_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return agg_df

    def plot_throughput_line(self):
        """
        Plot average throughput as line chart for different schedulers.
        """
        df = self.get_data()

        # Group by scheduler_type and run_num, then aggregate
        agg_df = (df
                  .groupby(["scheduler_type", "run_num"])
                  .agg({
                      "seq": "max",
                      "tokens_per_sec": "mean"
                  })
                  .reset_index()
                )

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting Throughput Line Chart...\n\n")

        # Create the line chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each scheduler type
        for scheduler in agg_df["scheduler_type"].unique():
            scheduler_data = agg_df[agg_df["scheduler_type"] == scheduler].sort_values("seq")
            ax.plot(
                scheduler_data["seq"],
                scheduler_data["tokens_per_sec"],
                marker='s',
                label=scheduler,
                linewidth=2.5,
                markersize=10
            )

        # Customize the plot
        ax.set_xlabel("Total Requests (max seq)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Throughput (tokens/sec)", fontsize=12, fontweight='bold')
        ax.set_title("Throughput Trend Across Request Loads", fontsize=14, fontweight='bold')
        ax.legend(title="Scheduler Type", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "throughput_line_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return agg_df

    def plot_latency_boxplot(self):
        """
        Plot box plot showing latency distribution (P50, P95, P99) for each scheduler.
        Box plots show the distribution and outliers of latency values.
        """
        df = self.get_data()

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting Latency Box Plot...\n\n")

        # Create figure with subplots for each request count
        request_counts = sorted(df.groupby("run_num")["seq"].max().unique())

        fig, axes = plt.subplots(1, len(request_counts), figsize=(16, 6), sharey=True)

        if len(request_counts) == 1:
            axes = [axes]

        for idx, req_count in enumerate(request_counts):
            # Filter data for this request count
            run_nums = df.groupby("run_num")["seq"].max()
            matching_runs = run_nums[run_nums == req_count].index
            subset = df[df["run_num"].isin(matching_runs)]

            # Prepare data for box plot
            schedulers = sorted(subset["scheduler_type"].unique())
            data_to_plot = [subset[subset["scheduler_type"] == scheduler]["latency"].values
                           for scheduler in schedulers]

            # Create box plot
            bp = axes[idx].boxplot(data_to_plot, labels=schedulers, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(schedulers)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[idx].set_title(f"{req_count} Requests", fontweight='bold')
            axes[idx].set_xlabel("Scheduler Type", fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=15)

        axes[0].set_ylabel("Latency (seconds)", fontsize=12, fontweight='bold')
        fig.suptitle("Latency Distribution by Scheduler Type and Request Count",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "latency_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

    def plot_latency_percentiles(self):
        """
        Plot latency percentiles (P50, P95, P99) for each scheduler across request counts.
        Shows how tail latencies vary with load.
        """
        df = self.get_data()

        # Status for plotting
        print ("*" * 60)
        print ("Start Plotting Latency Percentiles...\n\n")

        # Calculate percentiles for each scheduler and request count
        percentiles_df = (df
                         .groupby(["scheduler_type", "run_num"])
                         .agg({
                             "seq": "max",
                             "latency": lambda x: [
                                 np.percentile(x, 50),  # P50 (median)
                                 np.percentile(x, 95),  # P95
                                 np.percentile(x, 99)   # P99
                             ]
                         })
                         .reset_index()
                        )

        # Expand the percentile list into separate columns
        percentiles_df[['P50', 'P95', 'P99']] = pd.DataFrame(
            percentiles_df['latency'].tolist(),
            index=percentiles_df.index
        )

        # Create subplots for each percentile
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        percentiles = ['P50', 'P95', 'P99']

        for idx, percentile in enumerate(percentiles):
            # Pivot for this percentile
            pivot_df = percentiles_df.pivot(
                index="seq",
                columns="scheduler_type",
                values=percentile
            )

            # Set up bar positions
            x = np.arange(len(pivot_df.index))
            width = 0.25
            schedulers = pivot_df.columns.tolist()

            # Plot bars
            for i, scheduler in enumerate(schedulers):
                offset = width * (i - len(schedulers)/2 + 0.5)
                bars = axes[idx].bar(
                    x + offset,
                    pivot_df[scheduler],
                    width,
                    label=scheduler,
                    alpha=0.8
                )

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        axes[idx].text(
                            bar.get_x() + bar.get_width()/2.,
                            height,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom',
                            fontsize=7
                        )

            # Customize subplot
            axes[idx].set_xlabel("Total Requests", fontsize=10, fontweight='bold')
            axes[idx].set_ylabel(f"{percentile} Latency (seconds)", fontsize=10, fontweight='bold')
            axes[idx].set_title(f"{percentile} Latency", fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(pivot_df.index)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle("Latency Percentiles Across Request Loads",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save and show
        output_path = os.path.join(self.output_dir, "latency_percentiles.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()

        return percentiles_df

if __name__ == "__main__":
    benchmark_plot = BenchmarkPlot()

    # Bar charts
    print("\n=== Generating Bar Charts ===")
    benchmark_plot.plot_ttft()
    benchmark_plot.plot_throughput()
    benchmark_plot.plot_total_latency()

    # Line charts
    print("\n=== Generating Line Charts ===")
    benchmark_plot.plot_ttft_line()
    benchmark_plot.plot_throughput_line()

    # Box plots and percentiles
    print("\n=== Generating Distribution Charts ===")
    benchmark_plot.plot_latency_boxplot()
    benchmark_plot.plot_latency_percentiles()

    print("\n" + "="*60)
    print("All plots saved to 'plots/' folder!")
    print("="*60)
