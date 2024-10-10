import java.util.*;

public class Main {
    public static int maxProfit(int[] start, int[] end, int[] profit) {
        int n = start.length;

        // 按项目的结束时间排序，保持start、end、profit同步排序
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // 根据结束时间对项目进行排序
        Arrays.sort(indices, (i1, i2) -> Integer.compare(end[i1], end[i2]));

        // 动态规划数组
        int[] dp = new int[n];

        // 初始化第一个项目的最大收益
        dp[0] = profit[indices[0]];

        // 计算每个项目的最大收益
        for (int i = 1; i < n; i++) {
            // 当前项目的索引
            int currentIndex = indices[i];

            // 不选当前项目的收益
            int includeProfit = profit[currentIndex];

            // 使用 binarySearch 查找不冲突的项目
            int lastNonConflictingIndex = Arrays.binarySearch(end, 0, i, start[currentIndex]);
            if (lastNonConflictingIndex < 0) {
                // 如果没有精确匹配，返回的是一个负数，表示插入点位置。
                // 我们需要找到最近的非冲突项目，因此取 `-(lastNonConflictingIndex + 1) - 1`
                lastNonConflictingIndex = -(lastNonConflictingIndex + 1) - 1;
            }

            if (lastNonConflictingIndex != -1) {
                includeProfit += dp[lastNonConflictingIndex];
            }

            // 状态转移方程，选择不冲突的最大收益
            dp[i] = Math.max(dp[i - 1], includeProfit);
        }

        // 返回最大的收益
        return dp[n - 1];
    }

    public static void main(String[] args) {
        int[] start = { 1, 2, 3, 3 };
        int[] end = { 3, 4, 5, 6 };
        int[] profit = { 50, 10, 40, 70 };

        System.out.println("最大收益: " + maxProfit(start, end, profit)); // 输出: 120
    }
}
