// Exey Panteleev
import ArgumentParser
import Foundation

// MARK: - Ask Command

struct AskCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ask",
        abstract: "Ask questions about your codebase using AI"
    )

    @Argument(help: "The question to ask about the codebase")
    var question: String

    @Option(name: .long, help: "Path to the repository")
    var path: String = "."

    @Flag(name: .long, help: "Agentic retrieval: let the model pick files and read their contents (slower, sharper)")
    var deep: Bool = false

    func run() async throws {
        let config = ConfigLoader.load()

        guard config.ai.enabled, !config.ai.apiKey.isEmpty else {
            print("❌ AI features disabled. Enable them in .codecontext.json and set ai.apiKey")
            return
        }

        print("🤖 Analyzing codebase to answer: \"\(question)\"")
        print("   Gathering context...")

        let result = try await AnalysisPipeline.run(path: path, config: config)

        let hotspots = result.graph.getTopHotspots(limit: 10).map(\.path)

        let root = URL(fileURLWithPath: path).standardizedFileURL.path
        let files = result.parsedFiles
            .sorted { (result.graph.pageRankScores[$0.filePath] ?? 0) > (result.graph.pageRankScores[$1.filePath] ?? 0) }
            .map { file in
                FileDigest(
                    path: file.filePath.hasPrefix(root + "/")
                        ? String(file.filePath.dropFirst(root.count + 1))
                        : file.filePath,
                    absolutePath: file.filePath,
                    types: file.declarations.map(\.name),
                    lineCount: file.lineCount
                )
            }

        let context = CodebaseContext(
            totalFiles: result.parsedFiles.count,
            languages: ["Swift"],
            hotspots: hotspots,
            files: files
        )

        let aiAnalyzer = AICodeAnalyzer(
            apiKey: config.ai.apiKey,
            model: config.ai.model,
            provider: config.ai.provider,
            baseURL: config.ai.baseURL,
            tokenURL: config.ai.tokenURL
        )

        let response = deep
            ? try await aiAnalyzer.askQuestionDeep(question, context: context) { print("   ↳ \($0)") }
            : try await aiAnalyzer.askQuestion(question, context: context)

        print("\n💡 \(response.answer)\n")

        if !response.suggestedFiles.isEmpty {
            print("📁 Check these files:")
            for file in response.suggestedFiles {
                print("   - \(file)")
            }
        }

        print("\n🎯 Confidence: \(Int(response.confidence * 100))%")
    }
}
