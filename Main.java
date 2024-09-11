import java.util.*;

public class Main {

    static List<List<Integer>> tree; // 树的邻接表表示
    static int[] childCount; // 存储每个节点的子节点数量 public static void main(String[] args) {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("输入节点数量 n:");
        int n = scanner.nextInt();
        tree = new ArrayList<>();
        childCount = new int[n + 1];

        for (int i = 0; i <= n; i++) {
            tree.add(new ArrayList<>());
        }

        System.out.println("输入 n-1 个边，每行两个数 u 和 v，表示 v 是 u 的子节点:");
        for (int i = 1; i < n; i++) {
            int u = scanner.nextInt();
            int v = scanner.nextInt();
            tree.get(u).add(v); // 假设输入已确保 u 是父节点
        }

        // 计算每个节点的子节点数量
        dfs(1); // 假设根节点为 1

        // 根据子节点数量分组
        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            groups.computeIfAbsent(childCount[i], k -> new ArrayList<>()).add(i);
        }

        // 输出分组结果
        for (Map.Entry<Integer, List<Integer>> entry : groups.entrySet()) {
            System.out.println("子节点数 " + entry.getKey() + ": " + entry.getValue());
        }

        scanner.close();
    }

    // 深度优先搜索计算子节点数
    static void dfs(int node) {
        int count = 0;
        for (int child : tree.get(node)) {
            dfs(child);
            count += 1 + childCount[child]; // 子节点本身加上其子节点的总数
        }
        childCount[node] = count;
    }

}
