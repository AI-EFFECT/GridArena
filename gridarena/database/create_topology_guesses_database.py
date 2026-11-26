"""Creates the sql database tables for the user topology guess"""

from gridarena.database.rename_guess_column import rename_user_id_to_guess_id


def create_topology_guesses_database(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "TopologyUserGuesses" (
            guess_id TEXT NOT NULL,
            grid_id TEXT NOT NULL,
            guessed_topology TEXT NOT NULL,
            timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT topologyuserguesses_pk
                PRIMARY KEY (guess_id, grid_id),
            CONSTRAINT topologyuserguesses_grid_fk
                FOREIGN KEY (grid_id)
                    REFERENCES grids(grid_id) ON DELETE CASCADE
        );
        """
    )
    rename_user_id_to_guess_id(cursor, "TopologyUserGuesses")
