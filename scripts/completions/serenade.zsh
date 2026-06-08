#compdef serenade
# serenade zsh completion
#
# Install: source this file from your ~/.zshrc, e.g.
#   fpath=(/path/to/serenade/scripts/completions $fpath)
#   autoload -U compinit && compinit
# OR source it directly:
#   source /path/to/serenade/scripts/completions/serenade.zsh

_serenade() {
    local context state line
    typeset -A opt_args

    local -a flags
    flags=(
        '--setup[Guided first-run setup (writes .env)]'
        '--web[Spin up the stack and open the web UI]'
        '--down[Stop the stack]'
        '--volumes[(with --down) also remove volumes — destroys local data]'
        '--logs[Tail compose logs]:service:(orchestrator postgres redis neo4j minio opa worker)'
        '--status[Container + LLM health check]'
        '--seed[Seed the bundled DVWA demo engagement]'
        '--mlx[Run mlx_lm.server on the host]'
        '--mlx-model[MLX model id]:model:'
        '--mlx-port[MLX port]:port:'
        '--migrate[Apply alembic migrations]'
        '--test[Run pytest at a given scope]:scope:(unit integration e2e llm all)'
    )

    local -a subcommands
    subcommands=(
        'engagement:engagement lifecycle observability commands'
    )

    _arguments -C \
        $flags \
        '1: :->cmd' \
        '*::arg:->args'

    case "$state" in
        cmd)
            _describe -t commands 'serenade commands' subcommands
            ;;
        args)
            case "$words[1]" in
                engagement)
                    _serenade_engagement
                    ;;
            esac
            ;;
    esac
}

_serenade_engagement() {
    local -a eng_subcommands
    eng_subcommands=(
        'watch:stream high-signal events for an engagement'
        'summary:one-shot engagement digest'
    )

    _arguments -C \
        '1: :->subcmd' \
        '*::arg:->subargs'

    case "$state" in
        subcmd)
            _describe -t commands 'engagement subcommands' eng_subcommands
            ;;
        subargs)
            case "$words[1]" in
                watch)
                    _arguments \
                        '1:engagement_id:' \
                        '--filter[event-kind filter (comma-separated)]:kinds:(tool.completed tool.suppressed approval.queued approval.decided regression.requested regression.applied regression.vetoed stage.changed engagement.aborted agent.escalate_human engagement.budget_warning engagement.veto_rate_warning engagement.suppression_spread)' \
                        '--url[orchestrator base URL]:url:'
                    ;;
                summary)
                    _arguments \
                        '1:engagement_id:' \
                        '--format[output format]:format:(text json yaml)' \
                        '--url[orchestrator base URL]:url:'
                    ;;
            esac
            ;;
    esac
}

_serenade "$@"
