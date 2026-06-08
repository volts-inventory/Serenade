# serenade bash completion
#
# Install: source this file from your ~/.bashrc, e.g.
#   source /path/to/serenade/scripts/completions/serenade.bash
#
# Provides completion for the engagement watch/summary subcommands
# and the flag-based primary modes (--web, --setup, --status, etc.).

_serenade_complete() {
    local cur prev words cword
    _init_completion 2>/dev/null || {
        # Minimal fallback when bash-completion isn't loaded.
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
    }

    local flags="--setup --web --down --volumes --logs --status --seed --mlx --mlx-model --mlx-port --migrate --test"
    local subcmds="engagement"
    local eng_subcmds="watch summary"
    local eng_kinds="tool.completed tool.suppressed approval.queued approval.decided regression.requested regression.applied regression.vetoed stage.changed engagement.aborted agent.escalate_human engagement.budget_warning engagement.veto_rate_warning engagement.suppression_spread"
    local fmts="text json yaml"

    # Walk the words to figure out our context.
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "$subcmds $flags" -- "$cur") )
        return 0
    fi

    case "${COMP_WORDS[1]}" in
        engagement)
            if [[ ${COMP_CWORD} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "$eng_subcmds" -- "$cur") )
                return 0
            fi
            case "${COMP_WORDS[2]}" in
                watch)
                    case "$prev" in
                        --filter)
                            COMPREPLY=( $(compgen -W "$eng_kinds" -- "$cur") )
                            return 0
                            ;;
                        --url)
                            return 0
                            ;;
                    esac
                    if [[ "$cur" == --* ]]; then
                        COMPREPLY=( $(compgen -W "--filter --url" -- "$cur") )
                    fi
                    return 0
                    ;;
                summary)
                    case "$prev" in
                        --format)
                            COMPREPLY=( $(compgen -W "$fmts" -- "$cur") )
                            return 0
                            ;;
                        --url)
                            return 0
                            ;;
                    esac
                    if [[ "$cur" == --* ]]; then
                        COMPREPLY=( $(compgen -W "--format --url" -- "$cur") )
                    fi
                    return 0
                    ;;
            esac
            ;;
        *)
            # Default: complete flags + their value choices.
            case "$prev" in
                --test)
                    COMPREPLY=( $(compgen -W "unit integration e2e llm all" -- "$cur") )
                    return 0
                    ;;
                --logs)
                    COMPREPLY=( $(compgen -W "orchestrator postgres redis neo4j minio opa worker" -- "$cur") )
                    return 0
                    ;;
            esac
            COMPREPLY=( $(compgen -W "$flags" -- "$cur") )
            return 0
            ;;
    esac
}

complete -F _serenade_complete serenade
