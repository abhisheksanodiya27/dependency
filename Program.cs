using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.MSBuild;
using Newtonsoft.Json;

namespace CSharpDependencyAnalyzer
{
    class Program
    {
        static async Task Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: CSharpDependencyAnalyzer <project-path> <target-cs-file>");
                Console.WriteLine("Example: CSharpDependencyAnalyzer MyProject.csproj Services/UserService.cs");
                return;
            }

            string projectPath = args[0];
            string targetFile = args[1];

            try
            {
                MSBuildLocator.RegisterDefaults();
                var analyzer = new DependencyAnalyzer();
                var result = await analyzer.AnalyzeDependencies(projectPath, targetFile);
                
                string json = JsonConvert.SerializeObject(result, Formatting.Indented);
                Console.WriteLine(json);
            }
            catch (Exception ex)
            {
                var error = new
                {
                    error = true,
                    message = ex.Message,
                    stackTrace = ex.StackTrace
                };
                Console.WriteLine(JsonConvert.SerializeObject(error, Formatting.Indented));
                Environment.Exit(1);
            }
        }
    }

    public class DependencyAnalyzer
    {
        public async Task<DependencyResult> AnalyzeDependencies(string projectPath, string targetFilePath)
        {
            var workspace = MSBuildWorkspace.Create();
            var project = await workspace.OpenProjectAsync(projectPath);
            
            string fullTargetPath = Path.GetFullPath(targetFilePath);
            
            var document = project.Documents.FirstOrDefault(d => 
                Path.GetFullPath(d.FilePath ?? "") == fullTargetPath ||
                d.FilePath?.EndsWith(targetFilePath.Replace('/', '\\')) == true ||
                d.FilePath?.EndsWith(targetFilePath.Replace('\\', '/')) == true);

            if (document == null)
            {
                throw new FileNotFoundException($"File '{targetFilePath}' not found in project");
            }

            var compilation = await project.GetCompilationAsync();
            var syntaxTree = await document.GetSyntaxTreeAsync();
            var semanticModel = await document.GetSemanticModelAsync();
            var root = await syntaxTree.GetRootAsync();

            var result = new DependencyResult
            {
                TargetFile = targetFilePath,
                FullPath = document.FilePath,
                ProjectName = project.Name,
                Dependencies = new DependencyInfo()
            };

            // 1. EXPLICIT USING DIRECTIVES
            var usingDirectives = root.DescendantNodes().OfType<UsingDirectiveSyntax>();
            result.Dependencies.UsingDirectives = usingDirectives
                .Select(u => u.Name?.ToString())
                .Where(n => !string.IsNullOrEmpty(n))
                .Distinct()
                .OrderBy(n => n)
                .ToList();

            // 2. GLOBAL USINGS (from project-level or other files)
            var globalUsings = compilation.SyntaxTrees
                .SelectMany(tree => tree.GetRoot().DescendantNodes().OfType<UsingDirectiveSyntax>())
                .Where(u => u.GlobalKeyword.Kind() != SyntaxKind.None)
                .Select(u => u.Name?.ToString())
                .Where(n => !string.IsNullOrEmpty(n))
                .Distinct()
                .ToList();
            result.Dependencies.GlobalUsings = globalUsings;

            // 3. ALL TYPE REFERENCES (with or without using)
            var typeReferences = new Dictionary<string, TypeReferenceInfo>();
            var referencedFiles = new Dictionary<string, FileReferenceInfo>();
            var methodInvocations = new List<MethodInvocationInfo>();

            // Get all syntax nodes that could reference types
            var allNodes = root.DescendantNodes().ToList();

            // IDENTIFIERS (variables, parameters, etc.)
            foreach (var identifier in allNodes.OfType<IdentifierNameSyntax>())
            {
                AnalyzeSymbol(identifier, semanticModel, typeReferences, referencedFiles, 
                    document.FilePath, "Identifier");
            }

            // MEMBER ACCESS (MyClass.Method, obj.Property)
            foreach (var memberAccess in allNodes.OfType<MemberAccessExpressionSyntax>())
            {
                AnalyzeSymbol(memberAccess, semanticModel, typeReferences, referencedFiles, 
                    document.FilePath, "MemberAccess");
            }

            // OBJECT CREATION (new MyClass())
            foreach (var objectCreation in allNodes.OfType<ObjectCreationExpressionSyntax>())
            {
                AnalyzeSymbol(objectCreation, semanticModel, typeReferences, referencedFiles, 
                    document.FilePath, "ObjectCreation");
            }

            // METHOD INVOCATIONS
            foreach (var invocation in allNodes.OfType<InvocationExpressionSyntax>())
            {
                var symbolInfo = semanticModel.GetSymbolInfo(invocation);
                var symbol = symbolInfo.Symbol as IMethodSymbol;
                
                if (symbol != null && symbol.ContainingType != null)
                {
                    var containingType = symbol.ContainingType;
                    var sourceFile = GetSourceFile(containingType);
                    
                    if (!string.IsNullOrEmpty(sourceFile) && sourceFile != document.FilePath)
                    {
                        methodInvocations.Add(new MethodInvocationInfo
                        {
                            MethodName = symbol.Name,
                            FullMethodSignature = symbol.ToDisplayString(),
                            ContainingType = containingType.ToDisplayString(),
                            SourceFile = sourceFile,
                            IsStatic = symbol.IsStatic,
                            IsExtensionMethod = symbol.IsExtensionMethod
                        });

                        // Track the type reference
                        AnalyzeTypeSymbol(containingType, typeReferences, referencedFiles, 
                            document.FilePath, "MethodInvocation");
                    }
                }
            }

            // BASE TYPES (inheritance, interfaces)
            foreach (var baseType in allNodes.OfType<BaseTypeSyntax>())
            {
                var typeInfo = semanticModel.GetTypeInfo(baseType.Type);
                if (typeInfo.Type is INamedTypeSymbol namedType)
                {
                    AnalyzeTypeSymbol(namedType, typeReferences, referencedFiles, 
                        document.FilePath, "BaseType");
                }
            }

            // ATTRIBUTES
            foreach (var attribute in allNodes.OfType<AttributeSyntax>())
            {
                var symbolInfo = semanticModel.GetSymbolInfo(attribute);
                if (symbolInfo.Symbol?.ContainingType != null)
                {
                    AnalyzeTypeSymbol(symbolInfo.Symbol.ContainingType, typeReferences, 
                        referencedFiles, document.FilePath, "Attribute");
                }
            }

            // GENERIC TYPE ARGUMENTS
            foreach (var genericName in allNodes.OfType<GenericNameSyntax>())
            {
                var typeInfo = semanticModel.GetTypeInfo(genericName);
                if (typeInfo.Type is INamedTypeSymbol namedType)
                {
                    // Analyze the generic type itself
                    AnalyzeTypeSymbol(namedType, typeReferences, referencedFiles, 
                        document.FilePath, "GenericType");
                    
                    // Analyze type arguments
                    foreach (var typeArg in namedType.TypeArguments)
                    {
                        if (typeArg is INamedTypeSymbol argType)
                        {
                            AnalyzeTypeSymbol(argType, typeReferences, referencedFiles, 
                                document.FilePath, "GenericTypeArgument");
                        }
                    }
                }
            }

            // 4. PARTIAL CLASS DEPENDENCIES
            var partialClassFiles = new List<string>();
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            
            foreach (var classDecl in classDeclarations)
            {
                if (classDecl.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
                {
                    var symbol = semanticModel.GetDeclaredSymbol(classDecl) as INamedTypeSymbol;
                    if (symbol != null)
                    {
                        foreach (var location in symbol.Locations)
                        {
                            if (location.IsInSource)
                            {
                                string sourceFile = location.SourceTree?.FilePath;
                                if (!string.IsNullOrEmpty(sourceFile) && 
                                    sourceFile != document.FilePath &&
                                    !partialClassFiles.Contains(sourceFile))
                                {
                                    partialClassFiles.Add(sourceFile);
                                }
                            }
                        }
                    }
                }
            }

            result.Dependencies.PartialClassFiles = partialClassFiles
                .Select(f => new FileReference
                {
                    FullPath = f,
                    RelativePath = GetRelativePath(Path.GetDirectoryName(projectPath), f)
                })
                .ToList();

            // 5. SAME NAMESPACE FILES (implicit dependencies - no using needed)
            var sameNamespaceFiles = await GetSameNamespaceDependencies(
                project, document, semanticModel, root);

            result.Dependencies.SameNamespaceFiles = sameNamespaceFiles
                .Where(f => !referencedFiles.ContainsKey(f.FullPath)) // Don't duplicate
                .ToList();

            // Check if types from same-namespace files are actually used
            var implicitlyUsedFiles = new List<ImplicitFileUsage>();

            foreach (var nsFile in sameNamespaceFiles)
            {
                var otherDoc = project.Documents.FirstOrDefault(d => d.FilePath == nsFile.FullPath);
                if (otherDoc == null) continue;
                
                var otherSemanticModel = await otherDoc.GetSemanticModelAsync();
                var otherRoot = await (await otherDoc.GetSyntaxTreeAsync()).GetRootAsync();
                
                // Get all types declared in that file
                var declaredTypes = otherRoot.DescendantNodes()
                    .OfType<BaseTypeDeclarationSyntax>()
                    .Select(t => otherSemanticModel.GetDeclaredSymbol(t))
                    .Where(s => s != null)
                    .Select(s => s.ToDisplayString())
                    .ToList();
                
                // Check if any of these types are referenced in current file
                var usedTypes = declaredTypes
                    .Where(dt => typeReferences.ContainsKey(dt))
                    .ToList();
                
                if (usedTypes.Any())
                {
                    implicitlyUsedFiles.Add(new ImplicitFileUsage
                    {
                        FullPath = nsFile.FullPath,
                        RelativePath = nsFile.RelativePath,
                        UsedTypes = usedTypes,
                        Reason = "SameNamespace"
                    });
                }
            }

            result.Dependencies.ImplicitlyUsedFiles = implicitlyUsedFiles;

            // Populate results
            result.Dependencies.ReferencedTypes = typeReferences.Values.ToList();
            result.Dependencies.ReferencedFiles = referencedFiles.Values
                .OrderBy(f => f.RelativePath)
                .ToList();
            result.Dependencies.MethodInvocations = methodInvocations
                .OrderBy(m => m.ContainingType)
                .ThenBy(m => m.MethodName)
                .ToList();

            // Declared types
            result.Dependencies.DeclaredClasses = classDeclarations
                .Select(c => new TypeDeclaration
                {
                    Name = c.Identifier.Text,
                    Namespace = GetNamespace(c),
                    IsPublic = c.Modifiers.Any(m => m.IsKind(SyntaxKind.PublicKeyword)),
                    IsPartial = c.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)),
                    BaseTypes = c.BaseList?.Types.Select(t => t.Type.ToString()).ToList() ?? new List<string>()
                })
                .ToList();

            var interfaceDeclarations = root.DescendantNodes().OfType<InterfaceDeclarationSyntax>();
            result.Dependencies.DeclaredInterfaces = interfaceDeclarations
                .Select(i => new TypeDeclaration
                {
                    Name = i.Identifier.Text,
                    Namespace = GetNamespace(i),
                    IsPublic = i.Modifiers.Any(m => m.IsKind(SyntaxKind.PublicKeyword)),
                    IsPartial = false,
                    BaseTypes = i.BaseList?.Types.Select(t => t.Type.ToString()).ToList() ?? new List<string>()
                })
                .ToList();

            // External packages
            var externalNamespaces = result.Dependencies.UsingDirectives
                .Concat(result.Dependencies.GlobalUsings)
                .Where(ns => !ns.StartsWith(project.Name) && 
                            !ns.StartsWith("System.") && 
                            ns != "System")
                .Distinct()
                .ToList();

            result.Dependencies.ExternalPackages = externalNamespaces;

            return result;
        }

        private async Task<List<FileReference>> GetSameNamespaceDependencies(
            Project project, 
            Document currentDocument, 
            SemanticModel semanticModel,
            SyntaxNode root)
        {
            var sameNamespaceFiles = new List<FileReference>();
            
            // Get the namespace(s) of current file
            var currentNamespaces = root.DescendantNodes()
                .OfType<NamespaceDeclarationSyntax>()
                .Select(ns => ns.Name.ToString())
                .ToHashSet();
            
            var fileScopedNamespaces = root.DescendantNodes()
                .OfType<FileScopedNamespaceDeclarationSyntax>()
                .Select(ns => ns.Name.ToString())
                .ToHashSet();
            
            currentNamespaces.UnionWith(fileScopedNamespaces);
            
            if (!currentNamespaces.Any())
                return sameNamespaceFiles;
            
            // Check all other files in the project
            foreach (var otherDoc in project.Documents)
            {
                if (otherDoc.FilePath == currentDocument.FilePath)
                    continue;
                    
                var otherTree = await otherDoc.GetSyntaxTreeAsync();
                if (otherTree == null) continue;
                
                var otherRoot = await otherTree.GetRootAsync();
                
                // Get namespaces from other file
                var otherNamespaces = otherRoot.DescendantNodes()
                    .OfType<NamespaceDeclarationSyntax>()
                    .Select(ns => ns.Name.ToString())
                    .ToHashSet();
                
                var otherFileScopedNs = otherRoot.DescendantNodes()
                    .OfType<FileScopedNamespaceDeclarationSyntax>()
                    .Select(ns => ns.Name.ToString())
                    .ToHashSet();
                
                otherNamespaces.UnionWith(otherFileScopedNs);
                
                // Check if they share any namespace
                if (currentNamespaces.Intersect(otherNamespaces).Any())
                {
                    sameNamespaceFiles.Add(new FileReference
                    {
                        FullPath = otherDoc.FilePath,
                        RelativePath = GetRelativePath(
                            Path.GetDirectoryName(project.FilePath), 
                            otherDoc.FilePath
                        )
                    });
                }
            }
            
            return sameNamespaceFiles;
        }

        private void AnalyzeSymbol(SyntaxNode node, SemanticModel semanticModel, 
            Dictionary<string, TypeReferenceInfo> typeReferences, 
            Dictionary<string, FileReferenceInfo> referencedFiles,
            string currentFilePath, string referenceContext)
        {
            var symbolInfo = semanticModel.GetSymbolInfo(node);
            var symbol = symbolInfo.Symbol;

            if (symbol != null)
            {
                var containingType = symbol.ContainingType ?? 
                    (symbol as ITypeSymbol) as INamedTypeSymbol;
                
                if (containingType != null)
                {
                    AnalyzeTypeSymbol(containingType, typeReferences, referencedFiles, 
                        currentFilePath, referenceContext);
                }
            }
        }

        private void AnalyzeTypeSymbol(INamedTypeSymbol typeSymbol, 
            Dictionary<string, TypeReferenceInfo> typeReferences,
            Dictionary<string, FileReferenceInfo> referencedFiles,
            string currentFilePath, string referenceContext)
        {
            if (typeSymbol == null || typeSymbol.ContainingNamespace.IsGlobalNamespace)
                return;

            string fullTypeName = typeSymbol.ToDisplayString();
            
            // Add type reference
            if (!typeReferences.ContainsKey(fullTypeName))
            {
                typeReferences[fullTypeName] = new TypeReferenceInfo
                {
                    FullName = fullTypeName,
                    Namespace = typeSymbol.ContainingNamespace.ToDisplayString(),
                    AssemblyName = typeSymbol.ContainingAssembly?.Name,
                    IsFromSource = typeSymbol.Locations.Any(l => l.IsInSource),
                    ReferenceContexts = new List<string>()
                };
            }
            
            if (!typeReferences[fullTypeName].ReferenceContexts.Contains(referenceContext))
            {
                typeReferences[fullTypeName].ReferenceContexts.Add(referenceContext);
            }

            // Add source file reference
            var sourceFile = GetSourceFile(typeSymbol);
            if (!string.IsNullOrEmpty(sourceFile) && sourceFile != currentFilePath)
            {
                if (!referencedFiles.ContainsKey(sourceFile))
                {
                    referencedFiles[sourceFile] = new FileReferenceInfo
                    {
                        FullPath = sourceFile,
                        RelativePath = Path.GetFileName(sourceFile),
                        ReferencedTypes = new List<string>(),
                        ReferenceCount = 0
                    };
                }
                
                if (!referencedFiles[sourceFile].ReferencedTypes.Contains(fullTypeName))
                {
                    referencedFiles[sourceFile].ReferencedTypes.Add(fullTypeName);
                }
                referencedFiles[sourceFile].ReferenceCount++;
            }
        }

        private string GetSourceFile(INamedTypeSymbol typeSymbol)
        {
            var location = typeSymbol.Locations.FirstOrDefault(l => l.IsInSource);
            return location?.SourceTree?.FilePath;
        }

        private string GetNamespace(SyntaxNode node)
        {
            var namespaceDecl = node.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();
            if (namespaceDecl != null)
                return namespaceDecl.Name.ToString();

            var fileScopedNamespace = node.Ancestors().OfType<FileScopedNamespaceDeclarationSyntax>().FirstOrDefault();
            if (fileScopedNamespace != null)
                return fileScopedNamespace.Name.ToString();

            return string.Empty;
        }

        private string GetRelativePath(string fromPath, string toPath)
        {
            if (string.IsNullOrEmpty(fromPath) || string.IsNullOrEmpty(toPath))
                return toPath;

            try
            {
                Uri fromUri = new Uri(fromPath.EndsWith(Path.DirectorySeparatorChar.ToString()) 
                    ? fromPath 
                    : fromPath + Path.DirectorySeparatorChar);
                Uri toUri = new Uri(toPath);

                Uri relativeUri = fromUri.MakeRelativeUri(toUri);
                return Uri.UnescapeDataString(relativeUri.ToString().Replace('/', Path.DirectorySeparatorChar));
            }
            catch
            {
                return toPath;
            }
        }
    }

    // JSON output models
    public class DependencyResult
    {
        [JsonProperty("targetFile")]
        public string TargetFile { get; set; }

        [JsonProperty("fullPath")]
        public string FullPath { get; set; }

        [JsonProperty("projectName")]
        public string ProjectName { get; set; }

        [JsonProperty("dependencies")]
        public DependencyInfo Dependencies { get; set; }
    }

    public class DependencyInfo
    {
        [JsonProperty("usingDirectives")]
        public List<string> UsingDirectives { get; set; } = new List<string>();

        [JsonProperty("globalUsings")]
        public List<string> GlobalUsings { get; set; } = new List<string>();

        [JsonProperty("referencedTypes")]
        public List<TypeReferenceInfo> ReferencedTypes { get; set; } = new List<TypeReferenceInfo>();

        [JsonProperty("referencedFiles")]
        public List<FileReferenceInfo> ReferencedFiles { get; set; } = new List<FileReferenceInfo>();

        [JsonProperty("methodInvocations")]
        public List<MethodInvocationInfo> MethodInvocations { get; set; } = new List<MethodInvocationInfo>();

        [JsonProperty("partialClassFiles")]
        public List<FileReference> PartialClassFiles { get; set; } = new List<FileReference>();

        [JsonProperty("sameNamespaceFiles")]
        public List<FileReference> SameNamespaceFiles { get; set; } = new List<FileReference>();

        [JsonProperty("implicitlyUsedFiles")]
        public List<ImplicitFileUsage> ImplicitlyUsedFiles { get; set; } = new List<ImplicitFileUsage>();

        [JsonProperty("declaredClasses")]
        public List<TypeDeclaration> DeclaredClasses { get; set; } = new List<TypeDeclaration>();

        [JsonProperty("declaredInterfaces")]
        public List<TypeDeclaration> DeclaredInterfaces { get; set; } = new List<TypeDeclaration>();

        [JsonProperty("externalPackages")]
        public List<string> ExternalPackages { get; set; } = new List<string>();
    }

    public class TypeReferenceInfo
    {
        [JsonProperty("fullName")]
        public string FullName { get; set; }

        [JsonProperty("namespace")]
        public string Namespace { get; set; }

        [JsonProperty("assemblyName")]
        public string AssemblyName { get; set; }

        [JsonProperty("isFromSource")]
        public bool IsFromSource { get; set; }

        [JsonProperty("referenceContexts")]
        public List<string> ReferenceContexts { get; set; }
    }

    public class FileReferenceInfo
    {
        [JsonProperty("fullPath")]
        public string FullPath { get; set; }

        [JsonProperty("relativePath")]
        public string RelativePath { get; set; }

        [JsonProperty("referencedTypes")]
        public List<string> ReferencedTypes { get; set; }

        [JsonProperty("referenceCount")]
        public int ReferenceCount { get; set; }
    }

    public class MethodInvocationInfo
    {
        [JsonProperty("methodName")]
        public string MethodName { get; set; }

        [JsonProperty("fullMethodSignature")]
        public string FullMethodSignature { get; set; }

        [JsonProperty("containingType")]
        public string ContainingType { get; set; }

        [JsonProperty("sourceFile")]
        public string SourceFile { get; set; }

        [JsonProperty("isStatic")]
        public bool IsStatic { get; set; }

        [JsonProperty("isExtensionMethod")]
        public bool IsExtensionMethod { get; set; }
    }

    public class FileReference
    {
        [JsonProperty("fullPath")]
        public string FullPath { get; set; }

        [JsonProperty("relativePath")]
        public string RelativePath { get; set; }
    }

    public class ImplicitFileUsage
    {
        [JsonProperty("fullPath")]
        public string FullPath { get; set; }
        
        [JsonProperty("relativePath")]
        public string RelativePath { get; set; }
        
        [JsonProperty("usedTypes")]
        public List<string> UsedTypes { get; set; }
        
        [JsonProperty("reason")]
        public string Reason { get; set; }
    }

    public class TypeDeclaration
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("namespace")]
        public string Namespace { get; set; }

        [JsonProperty("isPublic")]
        public bool IsPublic { get; set; }

        [JsonProperty("isPartial")]
        public bool IsPartial { get; set; }

        [JsonProperty("baseTypes")]
        public List<string> BaseTypes { get; set; } = new List<string>();
    }
}